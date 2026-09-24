#!/usr/bin/env bash
# Back up, restore, and verify a Docker installation of Morphik Core.
#
# Usage:
#   ./morphik-backup.sh backup [--include-env] [--output-dir DIR] [--no-upload]
#   ./morphik-backup.sh restore FILE [--force] [--restore-config] [--no-safety-backup] [--no-start]
#   ./morphik-backup.sh verify FILE
#   ./morphik-backup.sh list [--output-dir DIR]
#
# Run it from the Morphik install directory, next to docker-compose.run.yml. The host needs only
# bash and Docker. PostgreSQL commands run through `docker compose exec -T postgres`. Archive,
# checksum, and manifest work runs in a short-lived container from the PostgreSQL image, so the
# PostgreSQL client tools always match the server.
#
# The `scheduled` and `offsite-sync` commands are the entry points of the optional `backup` and
# `backup-s3` services in docker-compose.run.yml. They run this same file inside a container.
#
# A backup is one uncompressed tar file:
#   manifest.json         versions, configuration, counts, and a sha256 for every other part
#   database.dump         pg_dump -Fc of the morphik database, embeddings included
#   storage.tar           the local storage directory (absent when storage is S3)
#   config/morphik.toml   the configuration at backup time
#   config/.env           only with --include-env, because it holds secrets
#
# Backups contain customer data. Files and directories are created owner-only (umask 077), and
# the script never prints the contents of .env.

set -euo pipefail
umask 077

COMPOSE_FILE="${MORPHIK_COMPOSE_FILE:-docker-compose.run.yml}"
PG_USER="${MORPHIK_PG_USER:-morphik}"
PG_DB="${MORPHIK_PG_DATABASE:-morphik}"
DEFAULT_TOOL_IMAGE="pgvector/pgvector:pg16"
AWS_IMAGE="${MORPHIK_BACKUP_AWS_IMAGE:-amazon/aws-cli:latest}"
ARCHIVE_IN_TOOL="/backup/archive.backup"

SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Inside the `backup` and `backup-s3` services the install directory is mounted at /install and
# the backup directory at /backups. PostgreSQL is reached over the Compose network.
if [ "${MORPHIK_BACKUP_IN_CONTAINER:-}" = "1" ]; then
    IN_CONTAINER=1
    INSTALL_DIR="${MORPHIK_INSTALL_DIR:-/install}"
else
    IN_CONTAINER=0
    INSTALL_DIR="${MORPHIK_INSTALL_DIR:-$SCRIPT_DIR}"
fi
CONFIG_FILE="$INSTALL_DIR/morphik.toml"
if [ "$IN_CONTAINER" = "1" ]; then
    ENV_FILE="$INSTALL_DIR/.env"
    STORAGE_DIR="${MORPHIK_STORAGE_DIR:-$INSTALL_DIR/storage}"
else
    ENV_FILE="${MORPHIK_ENV_FILE:-.env}"
    STORAGE_DIR="${MORPHIK_STORAGE_DIR:-./storage}"
fi

CLEANUP_PATHS=()
PSQL_PID=""
RESTORE_STAGE=""
SAFETY_BACKUP=""
CREATED_BACKUP=""

info() { printf '%s\n' "$*" >&2; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    local status=$?
    exec 3>&- 4<&-
    if [ -n "$PSQL_PID" ]; then
        kill "$PSQL_PID" 2>/dev/null || true
    fi
    local path
    for path in ${CLEANUP_PATHS[@]+"${CLEANUP_PATHS[@]}"}; do
        rm -rf "$path" 2>/dev/null || true
    done
    if [ -n "$RESTORE_STAGE" ]; then
        restore_failure_note
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

utc_stamp() { date -u +%Y%m%dT%H%M%SZ; }
utc_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Sleep in the background so `docker compose stop` interrupts the scheduled loops at once.
pause() {
    sleep "$1" &
    wait $! || true
}

abs_path() {
    local path=$1
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$(cd "$(dirname "$path")" && pwd)" "$(basename "$path")" ;;
    esac
}

# Read one scalar from a TOML file: toml_get FILE SECTION KEY. Handles the flat
# `key = value` lines Morphik uses. Strings lose their quotes and trailing comments are dropped.
toml_get() {
    [ -f "$1" ] || return 0
    awk -v want_section="$2" -v want_key="$3" '
        function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t\r]+$/, "", s); return s }
        /^[ \t]*\[/ {
            header = $0
            sub(/#.*/, "", header)
            header = trim(header)
            gsub(/^\[+|\]+$/, "", header)
            section = trim(header)
            next
        }
        section == want_section {
            line = $0
            if (line ~ /^[ \t]*#/) next
            eq = index(line, "=")
            if (eq == 0) next
            key = trim(substr(line, 1, eq - 1))
            gsub(/^"|"$/, "", key)
            if (key != want_key) next
            value = trim(substr(line, eq + 1))
            if (value ~ /^"/) {
                value = substr(value, 2)
                value = substr(value, 1, index(value, "\"") - 1)
            } else if (value ~ /^\047/) {
                value = substr(value, 2)
                value = substr(value, 1, index(value, "\047") - 1)
            } else {
                sub(/[ \t]*#.*$/, "", value)
                value = trim(value)
            }
            print value
            exit
        }
    ' "$1"
}

# Resolve the model_name of a [registered_models] entry, for example
# `openai_embedding = { model_name = "text-embedding-3-small" }`.
toml_model_name() {
    [ -f "$1" ] || return 0
    [ -n "$2" ] || return 0
    awk -v want_key="$2" '
        function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t\r]+$/, "", s); return s }
        /^[ \t]*\[/ {
            header = $0
            sub(/#.*/, "", header)
            header = trim(header)
            gsub(/^\[+|\]+$/, "", header)
            section = trim(header)
            next
        }
        section == "registered_models" {
            line = $0
            if (line ~ /^[ \t]*#/) next
            eq = index(line, "=")
            if (eq == 0) next
            key = trim(substr(line, 1, eq - 1))
            gsub(/^"|"$/, "", key)
            if (key != want_key) next
            rest = substr(line, eq + 1)
            if (match(rest, /model_name[ \t]*=[ \t]*"[^"]*"/)) {
                value = substr(rest, RSTART, RLENGTH)
                sub(/^model_name[ \t]*=[ \t]*"/, "", value)
                sub(/"$/, "", value)
                print value
            }
            exit
        }
    ' "$1"
}

# Print the configuration facts that decide whether a backup fits a deployment.
# Output is "type<TAB>path<TAB>value" lines for the manifest builder.
config_facts() {
    local toml=$1 model storage_provider
    model=$(toml_get "$toml" embedding model)
    storage_provider=$(toml_get "$toml" storage provider)
    fact s embedding.model "$model"
    fact s embedding.model_name "$(toml_model_name "$toml" "$model")"
    fact n embedding.dimensions "$(toml_get "$toml" embedding dimensions)"
    fact s embedding.similarity_metric "$(toml_get "$toml" embedding similarity_metric)"
    fact s vector_store.provider "$(toml_get "$toml" vector_store provider)"
    fact s multivector_store.provider "$(toml_get "$toml" multivector_store provider)"
    fact b multivector_store.colpali_enabled "$(toml_get "$toml" morphik enable_colpali)"
    fact s multivector_store.colpali_mode "$(toml_get "$toml" morphik colpali_mode)"
    fact s storage.provider "${storage_provider:-local}"
    fact s storage.storage_path "$(toml_get "$toml" storage storage_path)"
    fact s storage.bucket "$(toml_get "$toml" storage bucket_name)"
    fact s core.service_version "$(toml_get "$toml" service version)"
}

# Emit one manifest fact. Empty values become JSON null.
fact() {
    local type=$1 path=$2 value=${3-}
    value=$(printf '%s' "$value" | tr -d '\t\r\n')
    if [ -z "$value" ]; then
        printf 'z\t%s\t\n' "$path"
    else
        printf '%s\t%s\t%s\n' "$type" "$path" "$value"
    fi
}

# JSON handling runs in Perl because JSON::PP ships with the PostgreSQL image and the
# aws-cli image. Hosts do not need jq or Python.
json_tool() {
    perl -e "$PERL_JSON_TOOL" -- "$@"
}

IFS= read -r -d '' PERL_JSON_TOOL <<'PERL' || true
use strict;
use warnings;
use JSON::PP;

my $MAX_FORMAT_VERSION = 1;
my $pretty = JSON::PP->new->utf8->canonical->pretty;
my $compact = JSON::PP->new->utf8->canonical;

sub slurp {
    my ($file) = @_;
    open(my $fh, '<:raw', $file) or die "cannot read $file: $!\n";
    local $/;
    my $data = <$fh>;
    close($fh);
    return $data;
}

sub load_json {
    my ($file) = @_;
    my $data = eval { JSON::PP->new->utf8->decode(slurp($file)) };
    die "$file is not valid JSON: $@" if $@;
    return $data;
}

sub get_path {
    my ($data, $path) = @_;
    for my $key (split /\./, $path) {
        return undef unless ref($data) eq 'HASH' && exists $data->{$key};
        $data = $data->{$key};
    }
    return $data;
}

sub set_path {
    my ($data, $path, $value) = @_;
    my @keys = split /\./, $path;
    my $last = pop @keys;
    for my $key (@keys) {
        $data->{$key} = {} unless ref($data->{$key}) eq 'HASH';
        $data = $data->{$key};
    }
    $data->{$last} = $value;
}

sub scalar_text {
    my ($value) = @_;
    return '' unless defined $value;
    return $value ? 'true' : 'false' if JSON::PP::is_bool($value);
    return $compact->encode($value) if ref $value;
    return "$value";
}

sub table_rows {
    my ($manifest, $table) = @_;
    my $entry = get_path($manifest, "database.tables.$table");
    return undef unless ref($entry) eq 'HASH';
    return $entry->{rows} + 0;
}

sub build_manifest {
    my ($facts_file, $db_file) = @_;
    my %manifest;
    my @parts;
    open(my $fh, '<:raw', $facts_file) or die "cannot read $facts_file: $!\n";
    while (my $line = <$fh>) {
        chomp $line;
        next if $line eq '';
        my ($type, $path, @rest) = split /\t/, $line, -1;
        my $value = join("\t", @rest);
        if ($type eq 'p') {
            my ($name, $sha, $bytes) = @rest;
            push @parts, { name => $name, sha256 => $sha, bytes => $bytes + 0 };
            next;
        }
        if ($type eq 'z') { set_path(\%manifest, $path, undef); next }
        if ($type eq 's') { set_path(\%manifest, $path, $value); next }
        if ($type eq 'n') {
            die "fact $path is not a number: $value\n" unless $value =~ /^-?\d+(\.\d+)?$/;
            set_path(\%manifest, $path, $value + 0);
            next;
        }
        if ($type eq 'b') {
            set_path(\%manifest, $path, ($value eq 'true') ? JSON::PP::true : JSON::PP::false);
            next;
        }
        if ($type eq 'j') {
            set_path(\%manifest, $path, JSON::PP->new->utf8->allow_nonref->decode($value));
            next;
        }
        die "unknown fact type '$type' for $path\n";
    }
    close($fh);

    my $db = load_json($db_file);
    $manifest{format} = 'morphik-core-backup';
    $manifest{format_version} = $MAX_FORMAT_VERSION;
    $manifest{database} = {
        %{ $manifest{database} || {} },
        postgres_version => $db->{postgres_version},
        extensions => $db->{extensions} || {},
        vector_dimensions => $db->{vector_dimensions},
        documents_by_status => $db->{documents_by_status} || {},
        tables => $db->{tables} || {},
        schema => {
            migration_tracking => 'none',
            fingerprint => $db->{schema_fingerprint},
            note => 'Morphik Core creates and alters tables at startup and keeps no migration version. '
                . 'The fingerprint is an md5 of the public table, column, and type names.',
        },
    };
    $manifest{parts} = [ sort { $a->{name} cmp $b->{name} } @parts ];

    my %by_status = %{ $db->{documents_by_status} || {} };
    my $documents = 0;
    $documents += $_ for values %by_status;
    $manifest{counts} = {
        %{ $manifest{counts} || {} },
        documents => $documents,
        documents_by_status => \%by_status,
        chunks => table_rows(\%manifest, 'vector_embeddings'),
        multivector_chunks => table_rows(\%manifest, 'multi_vector_embeddings'),
        folders => table_rows(\%manifest, 'folders'),
    };
    return \%manifest;
}

sub check_manifest {
    my ($manifest) = @_;
    my @errors;
    push @errors, 'manifest.json is not a JSON object' unless ref($manifest) eq 'HASH';
    return @errors if @errors;
    my $format = $manifest->{format} // '';
    push @errors, "not a Morphik Core backup (format is '$format')" unless $format eq 'morphik-core-backup';
    my $version = $manifest->{format_version};
    if (!defined $version || $version !~ /^\d+$/) {
        push @errors, 'format_version is missing';
    } elsif ($version > $MAX_FORMAT_VERSION) {
        push @errors, "backup format $version is newer than this script supports ($MAX_FORMAT_VERSION). Download the current morphik-backup.sh.";
    }
    push @errors, 'created_at is missing' unless $manifest->{created_at};
    push @errors, 'database.tables is missing' unless ref(get_path($manifest, 'database.tables')) eq 'HASH';
    my $parts = $manifest->{parts};
    if (ref($parts) ne 'ARRAY') {
        push @errors, 'parts list is missing';
    } else {
        my %seen;
        for my $part (@$parts) {
            my $name = $part->{name} // '';
            push @errors, "part '$name' has an unsafe name" if $name eq '' || $name =~ m{(^/|\.\.)};
            push @errors, "part '$name' has no sha256" unless ($part->{sha256} // '') =~ /^[0-9a-f]{64}$/;
            $seen{$name} = 1;
        }
        push @errors, 'database.dump is missing from parts' unless $seen{'database.dump'};
        if (JSON::PP::is_bool(get_path($manifest, 'storage.included')) && get_path($manifest, 'storage.included')) {
            push @errors, 'storage.tar is missing from parts' unless $seen{'storage.tar'};
        }
    }
    return @errors;
}

sub read_target_facts {
    my ($file) = @_;
    my %facts;
    open(my $fh, '<:raw', $file) or die "cannot read $file: $!\n";
    while (my $line = <$fh>) {
        chomp $line;
        my ($type, $path, $value) = split /\t/, $line, 3;
        next unless defined $path;
        $facts{$path} = ($type eq 'z') ? undef : $value;
    }
    close($fh);
    return \%facts;
}

sub same_text {
    my ($left, $right, $default) = @_;
    $left = $default if !defined $left || $left eq '';
    $right = $default if !defined $right || $right eq '';
    return (defined $left ? $left : '') eq (defined $right ? $right : '');
}

# Decide whether a backup can be restored under the target configuration. Every error
# means the restored embeddings or files would not be usable as-is.
sub compat {
    my ($manifest, $target) = @_;
    my (@errors, @notes);

    my $backup_dims = get_path($manifest, 'database.vector_dimensions');
    $backup_dims = get_path($manifest, 'embedding.dimensions') unless defined $backup_dims;
    my $target_dims = $target->{'embedding.dimensions'};
    if (!defined $target_dims || $target_dims eq '') {
        push @errors, 'the target morphik.toml has no [embedding] dimensions';
    } elsif (defined $backup_dims && $backup_dims + 0 != $target_dims + 0) {
        push @errors, "embedding dimensions differ: backup has $backup_dims, target morphik.toml has $target_dims";
    }

    my $backup_model = get_path($manifest, 'embedding.model') // '';
    my $backup_name = get_path($manifest, 'embedding.model_name') // '';
    my $target_model = $target->{'embedding.model'} // '';
    my $target_name = $target->{'embedding.model_name'} // '';
    if ($backup_name ne '' && $target_name ne '') {
        if ($backup_name ne $target_name) {
            push @errors, "embedding model differs: backup used '$backup_name' ($backup_model), target uses '$target_name' ($target_model)";
        } elsif ($backup_model ne $target_model) {
            push @notes, "embedding model key changed from '$backup_model' to '$target_model'; both resolve to '$backup_name'";
        }
    } elsif ($backup_model ne $target_model) {
        push @errors, "embedding model differs: backup used '$backup_model', target uses '$target_model'";
    }

    my @providers = (
        [ 'vector_store.provider', 'pgvector', 'vector store' ],
        [ 'multivector_store.provider', 'postgres', 'multivector store' ],
        [ 'storage.provider', 'local', 'storage provider' ],
    );
    for my $provider (@providers) {
        my ($path, $default, $label) = @$provider;
        my $backup_value = get_path($manifest, $path);
        my $target_value = $target->{$path};
        next if same_text($backup_value, $target_value, $default);
        $backup_value = $default if !defined $backup_value || $backup_value eq '';
        $target_value = $default if !defined $target_value || $target_value eq '';
        push @errors, "$label differs: backup used '$backup_value', target uses '$target_value'";
    }

    if ((get_path($manifest, 'vector_store.provider') // 'pgvector') ne 'pgvector') {
        push @notes, 'the backup vector store is not pgvector, so its standard embeddings are not in this backup';
    }
    if ((get_path($manifest, 'multivector_store.provider') // 'postgres') ne 'postgres') {
        push @notes, 'the backup multivector store is not PostgreSQL, so ColPali embeddings are not in this backup';
    }
    my $storage_included = get_path($manifest, 'storage.included');
    if (JSON::PP::is_bool($storage_included) && !$storage_included) {
        push @notes, 'source files were not copied because storage is not local; they must still exist in the storage bucket';
    }
    return (\@errors, \@notes);
}

sub compare_stats {
    my ($manifest, $stats) = @_;
    my @errors;
    my $expected = get_path($manifest, 'database.tables') || {};
    my $actual = $stats->{tables} || {};
    for my $table (sort keys %$expected) {
        my $want = $expected->{$table};
        my $got = $actual->{$table};
        if (!defined $got) {
            push @errors, "table $table is missing after restore";
            next;
        }
        if ($want->{rows} != $got->{rows}) {
            push @errors, "table $table has $got->{rows} rows, expected $want->{rows}";
        } elsif (($want->{checksum} // '') ne ($got->{checksum} // '')) {
            push @errors, "table $table content checksum differs";
        }
    }
    for my $table (sort keys %$actual) {
        push @errors, "unexpected table $table after restore" unless exists $expected->{$table};
    }
    my $want_status = $compact->encode(get_path($manifest, 'database.documents_by_status') || {});
    my $got_status = $compact->encode($stats->{documents_by_status} || {});
    push @errors, "document status counts differ: expected $want_status, got $got_status" if $want_status ne $got_status;
    my $want_dims = get_path($manifest, 'database.vector_dimensions');
    my $got_dims = $stats->{vector_dimensions};
    if (defined $want_dims && (!defined $got_dims || $want_dims != $got_dims)) {
        push @errors, 'vector dimensions differ after restore';
    }
    return @errors;
}

# Files the database references but the storage archive does not contain.
sub missing_files {
    my ($refs_file, $listing_file, $storage_path) = @_;
    my %present;
    open(my $list, '<:raw', $listing_file) or die "cannot read $listing_file: $!\n";
    while (my $entry = <$list>) {
        chomp $entry;
        $entry =~ s{^\./}{};
        $present{$entry} = 1 if $entry ne '' && $entry !~ m{/$};
    }
    close($list);

    my @roots = ('/app/storage/', 'storage/');
    if (defined $storage_path && $storage_path ne '') {
        my $root = $storage_path;
        $root =~ s{^\./}{/app/};
        $root .= '/' unless $root =~ m{/$};
        unshift @roots, $root;
    }
    my @missing;
    open(my $refs, '<:raw', $refs_file) or die "cannot read $refs_file: $!\n";
    while (my $line = <$refs>) {
        chomp $line;
        next if $line eq '';
        my ($doc, $bucket, $key) = split /\t/, $line, 3;
        next unless defined $key && $key ne '';
        my @candidates = ($key);
        push @candidates, "$bucket/$key" if defined $bucket && $bucket ne '' && $bucket ne 'storage';
        my $found = 0;
        for my $candidate (@candidates) {
            $candidate =~ s{^\./}{};
            my @forms = ($candidate);
            for my $root (@roots) {
                push @forms, substr($candidate, length($root)) if index($candidate, $root) == 0;
            }
            for my $form (@forms) {
                if ($present{$form}) { $found = 1; last }
            }
            last if $found;
        }
        push @missing, "$doc\t$key" unless $found;
    }
    close($refs);
    return @missing;
}

sub summary {
    my ($manifest) = @_;
    my $counts = $manifest->{counts} || {};
    my $by_status = $counts->{documents_by_status} || {};
    my $status_text = join(', ', map { "$_=$by_status->{$_}" } sort keys %$by_status) || 'none';
    my @lines = (
        'created:        ' . scalar_text($manifest->{created_at}),
        'core image:     ' . scalar_text(get_path($manifest, 'core.image')) . ' ' . scalar_text(get_path($manifest, 'core.image_id')),
        'embedding:      ' . scalar_text(get_path($manifest, 'embedding.model')) . ' ('
            . scalar_text(get_path($manifest, 'embedding.model_name')) . '), '
            . scalar_text(get_path($manifest, 'database.vector_dimensions') // get_path($manifest, 'embedding.dimensions')) . ' dimensions',
        'documents:      ' . scalar_text($counts->{documents}) . " ($status_text)",
        'chunks:         ' . scalar_text($counts->{chunks}) . ' standard, ' . scalar_text($counts->{multivector_chunks} // 0) . ' multivector',
        'folders:        ' . scalar_text($counts->{folders} // 0),
    );
    my $included = get_path($manifest, 'storage.included');
    if (JSON::PP::is_bool($included) && $included) {
        push @lines, 'source files:   ' . scalar_text($counts->{storage_files}) . ' files, ' . scalar_text($counts->{storage_bytes}) . ' bytes';
    } else {
        push @lines, 'source files:   not included (storage provider ' . scalar_text(get_path($manifest, 'storage.provider')) . ')';
    }
    push @lines, 'includes .env:  ' . (($manifest->{includes_env} && $manifest->{includes_env} ne 'false') ? 'yes' : 'no');
    return @lines;
}

my $command = shift @ARGV // '';
if ($command eq 'manifest') {
    print $pretty->encode(build_manifest(@ARGV));
} elsif ($command eq 'get') {
    my ($file, $path) = @ARGV;
    my $value = get_path(load_json($file), $path);
    print scalar_text($value), "\n";
} elsif ($command eq 'parts') {
    for my $part (@{ load_json($ARGV[0])->{parts} || [] }) {
        print join("\t", $part->{name}, $part->{sha256}, $part->{bytes}), "\n";
    }
} elsif ($command eq 'check') {
    my @errors = check_manifest(load_json($ARGV[0]));
    print STDERR "manifest: $_\n" for @errors;
    exit(@errors ? 1 : 0);
} elsif ($command eq 'compat') {
    my ($errors, $notes) = compat(load_json($ARGV[0]), read_target_facts($ARGV[1]));
    print "NOTE: $_\n" for @$notes;
    print "INCOMPATIBLE: $_\n" for @$errors;
    exit(@$errors ? 1 : 0);
} elsif ($command eq 'compare') {
    my @errors = compare_stats(load_json($ARGV[0]), load_json($ARGV[1]));
    print "MISMATCH: $_\n" for @errors;
    exit(@errors ? 1 : 0);
} elsif ($command eq 'missing-files') {
    my @missing = missing_files(@ARGV);
    print "$_\n" for @missing;
    exit(@missing ? 1 : 0);
} elsif ($command eq 'summary') {
    print "$_\n" for summary(load_json($ARGV[0]));
} elsif ($command eq 'requeue-body') {
    my @ids;
    open(my $fh, '<:raw', $ARGV[0]) or die "cannot read $ARGV[0]: $!\n";
    while (my $id = <$fh>) {
        chomp $id;
        push @ids, { external_id => $id } if $id ne '';
    }
    print $pretty->encode({ jobs => \@ids });
} else {
    die "unknown json tool command '$command'\n";
}
PERL

# ---------------------------------------------------------------------------
# Docker and PostgreSQL access
# ---------------------------------------------------------------------------

compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

# Run a PostgreSQL client program against the deployment database.
pg() {
    if [ "$IN_CONTAINER" = "1" ]; then
        "$@"
    else
        compose exec -T postgres "$@"
    fi
}

require_host_deployment() {
    [ "$IN_CONTAINER" = "0" ] || return 0
    command -v docker >/dev/null 2>&1 || die "Docker is required."
    docker info >/dev/null 2>&1 || die "Docker is installed, but the daemon is not running."
    [ -f "$COMPOSE_FILE" ] || die "$COMPOSE_FILE not found in $INSTALL_DIR. Run this script from the Morphik install directory."
}

postgres_container() {
    compose ps -q postgres 2>/dev/null | head -n 1
}

require_running_postgres() {
    if [ "$IN_CONTAINER" = "0" ] && [ -z "$(postgres_container)" ]; then
        die "PostgreSQL is not running. Start Morphik with ./start-morphik.sh first."
    fi
    wait_for_postgres
}

wait_for_postgres() {
    local attempts=60
    while [ "$attempts" -gt 0 ]; do
        if pg pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts - 1))
        sleep 2
    done
    die "PostgreSQL did not become ready."
}

# The archive and file work runs in the same image as the PostgreSQL service so pg_restore
# and pg_dump always match the server version.
tool_image() {
    if [ -n "${MORPHIK_BACKUP_TOOL_IMAGE:-}" ]; then
        printf '%s\n' "$MORPHIK_BACKUP_TOOL_IMAGE"
        return
    fi
    local cid image=""
    cid=$(postgres_container 2>/dev/null || true)
    if [ -n "$cid" ]; then
        image=$(docker inspect -f '{{.Config.Image}}' "$cid" 2>/dev/null || true)
    fi
    printf '%s\n' "${image:-$DEFAULT_TOOL_IMAGE}"
}

# run_tool [docker run options...] -- internal-command [args...]
run_tool() {
    local options=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        options+=("$1")
        shift
    done
    shift
    docker run --rm -i --user 0 --network none \
        -e MORPHIK_BACKUP_TOOL=1 \
        -v "$SCRIPT_PATH:/opt/morphik/morphik-backup.sh:ro" \
        ${options[@]+"${options[@]}"} \
        --entrypoint bash "$TOOL_IMAGE" /opt/morphik/morphik-backup.sh "$@"
}

# ---------------------------------------------------------------------------
# Database snapshot, statistics, and fingerprints
# ---------------------------------------------------------------------------

# Session settings that make row text identical on every server that restores the dump.
STATS_SESSION_SQL="SET TIME ZONE 'UTC'; SET datestyle = 'ISO, YMD'; SET intervalstyle = 'postgres'; SET extra_float_digits = 1; SET bytea_output = 'hex';"

# Builds and runs (\gexec) one query that returns a single JSON line: server version,
# extensions, schema fingerprint, embedding dimensions, document status counts, and for every
# public table its row count and an order-independent checksum of the row text. The checksum
# covers embeddings, so a restore that changes a single vector fails the comparison.
IFS= read -r -d '' STATS_SQL <<'SQL' || true
SELECT format(
    $query$SELECT json_build_object(
        'postgres_version', current_setting('server_version'),
        'extensions', COALESCE((SELECT json_object_agg(extname, extversion ORDER BY extname) FROM pg_extension), '{}'::json),
        'schema_fingerprint', (
            SELECT md5(COALESCE(string_agg(c.table_name || '.' || c.column_name || ':' || c.udt_name, ',' ORDER BY c.table_name, c.column_name), ''))
            FROM information_schema.columns c
            WHERE c.table_schema = 'public'
        ),
        'vector_dimensions', (
            SELECT NULLIF(a.atttypmod, -1)
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass('public.vector_embeddings') AND a.attname = 'embedding' AND NOT a.attisdropped
        ),
        'documents_by_status', %s,
        'tables', %s
    )$query$,
    CASE
        WHEN to_regclass('public.documents') IS NULL THEN $$'{}'::json$$
        ELSE $$(SELECT COALESCE(json_object_agg(status, n ORDER BY status), '{}'::json) FROM (SELECT COALESCE(system_metadata->>'status', 'unknown') AS status, count(*) AS n FROM public.documents GROUP BY 1) s)$$
    END,
    COALESCE(
        (
            SELECT '(SELECT json_object_agg(name, json_build_object(''rows'', n, ''checksum'', h) ORDER BY name) FROM ('
                || string_agg(
                    format(
                        $table$SELECT %L::text AS name, count(*) AS n, COALESCE(sum(('x' || substr(md5(morphik_backup_row::text), 1, 15))::bit(60)::bigint), 0)::text AS h FROM public.%I morphik_backup_row$table$,
                        c.relname,
                        c.relname
                    ),
                    ' UNION ALL '
                    ORDER BY c.relname
                )
                || ') s)'
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r' AND NOT c.relispartition
        ),
        $$'{}'::json$$
    )
) \gexec
SQL

# Tables that hold rows in the target database, as "table=rows" words.
IFS= read -r -d '' TARGET_DATA_SQL <<'SQL' || true
SELECT COALESCE(string_agg(format('%s=%s', name, n), ' ' ORDER BY name), '')
FROM (
    SELECT c.relname AS name,
           (xpath('/row/n/text()', query_to_xml(format('SELECT count(*) AS n FROM public.%I', c.relname), false, true, '')))[1]::text::bigint AS n
    FROM pg_class c
    JOIN pg_namespace ns ON ns.oid = c.relnamespace
    WHERE ns.nspname = 'public' AND c.relkind = 'r'
) t
WHERE n > 0;
SQL

# Documents whose storage object must be in the archive. Rows still processing can point at a
# file that was replaced after the dump, so only settled documents are checked.
IFS= read -r -d '' REFERENCED_FILES_SQL <<'SQL' || true
SELECT external_id, COALESCE(storage_info->>'bucket', ''), storage_info->>'key'
FROM documents
WHERE COALESCE(storage_info->>'key', '') <> ''
  AND COALESCE(system_metadata->>'status', 'completed') IN ('completed', 'failed');
SQL

# Documents captured mid-ingestion would stay "processing" forever, because the queued job
# lives in Redis and Redis is not backed up. Mark them failed and bump ingestion_revision, the
# same fence POST /ingest/requeue uses: any stale job for the old revision is skipped as
# superseded, and requeue assigns the next revision and queues it.
IFS= read -r -d '' STUCK_DOCUMENTS_SQL <<'SQL' || true
WITH stuck AS (
    UPDATE documents
    SET system_metadata = (COALESCE(system_metadata, '{}'::jsonb) - 'progress') || jsonb_build_object(
        'status', 'failed',
        'error', 'Ingestion was still running when backup ' || :'backup_name' || ' was taken. Requeue this document to finish ingestion.',
        'ingestion_revision', CASE
            WHEN system_metadata->>'ingestion_revision' ~ '^[0-9]+$' THEN (system_metadata->>'ingestion_revision')::bigint + 1
            ELSE 1
        END,
        'updated_at', to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"')
    )
    WHERE system_metadata->>'status' = 'processing'
    RETURNING external_id
)
SELECT external_id FROM stuck ORDER BY external_id;
SQL

# Dump the database and collect statistics from one exported snapshot, so the manifest counts
# describe exactly the rows in the dump even while ingestion keeps writing.
dump_database() {
    local dump_file=$1 stats_file=$2 fifo_dir snapshot stats
    fifo_dir=$(mktemp -d "${TMPDIR:-/tmp}/morphik-backup-psql.XXXXXX")
    CLEANUP_PATHS+=("$fifo_dir")
    mkfifo "$fifo_dir/in" "$fifo_dir/out"

    pg psql -X -At -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" \
        <"$fifo_dir/in" >"$fifo_dir/out" 2>"$fifo_dir/err" &
    PSQL_PID=$!
    exec 3>"$fifo_dir/in"
    exec 4<"$fifo_dir/out"

    printf '%s\nBEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\nSELECT pg_export_snapshot();\n' "$STATS_SESSION_SQL" >&3
    if ! IFS= read -r snapshot <&4 || [ -z "$snapshot" ]; then
        cat "$fifo_dir/err" >&2 || true
        die "Could not open a PostgreSQL snapshot."
    fi

    if ! pg pg_dump -U "$PG_USER" -d "$PG_DB" -Fc --snapshot="$snapshot" >"$dump_file"; then
        die "pg_dump failed."
    fi

    printf '%s\n' "$STATS_SQL" >&3
    if ! IFS= read -r stats <&4 || [ -z "$stats" ]; then
        cat "$fifo_dir/err" >&2 || true
        die "Could not read database statistics."
    fi
    printf '%s\n' "$stats" >"$stats_file"

    printf 'COMMIT;\n\\q\n' >&3
    exec 3>&-
    wait "$PSQL_PID" || true
    PSQL_PID=""
    exec 4<&-
}

# Run the statistics query outside a snapshot (after a restore, or against a verify database).
database_stats() {
    printf '%s\n%s\n' "$STATS_SESSION_SQL" "$STATS_SQL" | "$@" psql -X -At -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB"
}

# ---------------------------------------------------------------------------
# Internal commands. They run inside the tool container, or directly in the backup service.
# ---------------------------------------------------------------------------

# Write storage.tar to stdout. Files that change while tar reads them only produce a warning:
# the database dump was taken first, so it cannot reference a file that tar missed.
internal_storage_tar() {
    local dir=$1 status=0
    [ -d "$dir" ] || die "Storage directory $dir does not exist."
    tar -C "$dir" --numeric-owner --exclude='./.morphik-restore-*' -cf - . || status=$?
    if [ "$status" -eq 1 ]; then
        warn "Some storage files changed while they were copied. The archive still has every file the database dump references."
        status=0
    fi
    return "$status"
}

# internal_package WORK_DIR: add checksums and storage totals to the facts, build the
# manifest, and write the final archive to stdout. Nothing is written into WORK_DIR, so it can
# be mounted read-only.
internal_package() {
    local work=$1 tmp part sha bytes
    tmp=$(mktemp -d)
    CLEANUP_PATHS+=("$tmp")
    cp "$work/facts.tsv" "$tmp/facts.tsv"

    local parts=(database.dump)
    [ -f "$work/storage.tar" ] && parts+=(storage.tar)
    [ -f "$work/config/morphik.toml" ] && parts+=(config/morphik.toml)
    [ -f "$work/config/.env" ] && parts+=(config/.env)

    for part in "${parts[@]}"; do
        sha=$(sha256sum "$work/$part" | awk '{print $1}')
        bytes=$(stat -c %s "$work/$part")
        printf 'p\t%s\t%s\t%s\t%s\n' "parts" "$part" "$sha" "$bytes" >>"$tmp/facts.tsv"
    done

    if [ -f "$work/storage.tar" ]; then
        tar -tvf "$work/storage.tar" | awk '
            substr($1, 1, 1) == "-" { files += 1; bytes += $3 }
            END { printf "n\tcounts.storage_files\t%d\nn\tcounts.storage_bytes\t%d\n", files, bytes }
        ' >>"$tmp/facts.tsv"
        printf 'b\tstorage.included\ttrue\n' >>"$tmp/facts.tsv"
    else
        printf 'b\tstorage.included\tfalse\n' >>"$tmp/facts.tsv"
    fi

    json_tool manifest "$tmp/facts.tsv" "$work/database.json" >"$tmp/manifest.json"
    json_tool check "$tmp/manifest.json"

    tar --numeric-owner --owner=0 --group=0 -cf - -C "$tmp" manifest.json -C "$work" "${parts[@]}"
}

# internal_validate ARCHIVE: check the manifest and every part checksum, then print the
# manifest to stdout.
internal_validate() {
    local archive=$1 tmp name sha bytes actual
    tmp=$(mktemp -d)
    CLEANUP_PATHS+=("$tmp")
    [ -f "$archive" ] || die "Backup file not found."
    tar -tf "$archive" >"$tmp/members" 2>"$tmp/tar.err" || {
        cat "$tmp/tar.err" >&2
        die "The backup file is not a readable tar archive. It may be truncated."
    }
    grep -qx 'manifest.json' "$tmp/members" || die "The backup has no manifest.json."
    tar -xOf "$archive" --occurrence=1 manifest.json >"$tmp/manifest.json"
    json_tool check "$tmp/manifest.json" || die "The backup manifest is invalid."

    json_tool parts "$tmp/manifest.json" >"$tmp/parts"
    while IFS=$'\t' read -r name sha bytes; do
        grep -qxF "$name" "$tmp/members" || die "The backup is missing $name."
        actual=$(tar -xOf "$archive" --occurrence=1 "$name" | sha256sum | awk '{print $1}')
        [ "$actual" = "$sha" ] || die "Checksum mismatch for $name. The backup is damaged."
        info "  ok  $name ($bytes bytes)"
    done <"$tmp/parts"
    cat "$tmp/manifest.json"
}

internal_extract() {
    tar -xOf "$1" --occurrence=1 "$2"
}

# internal_compat MANIFEST TOML: exit non-zero if the backup cannot be restored under TOML.
internal_compat() {
    local manifest=$1 toml=$2 tmp
    tmp=$(mktemp -d)
    CLEANUP_PATHS+=("$tmp")
    config_facts "$toml" >"$tmp/target.tsv"
    json_tool compat "$manifest" "$tmp/target.tsv"
}

internal_manifest() {
    json_tool manifest "$1" "$2"
}

# internal_restore_storage ARCHIVE STORAGE_DIR: extract into a staging directory first, so a
# failed extraction leaves the current files untouched, then swap the contents in.
internal_restore_storage() {
    local archive=$1 dir=$2 stage entry
    mkdir -p "$dir"
    stage="$dir/.morphik-restore-$$"
    rm -rf "$stage"
    mkdir "$stage"
    if ! tar -xOf "$archive" --occurrence=1 storage.tar | tar -x --numeric-owner -p -C "$stage" -f -; then
        rm -rf "$stage"
        die "Could not extract storage.tar."
    fi
    find "$dir" -mindepth 1 -maxdepth 1 ! -name "$(basename "$stage")" -exec rm -rf {} +
    for entry in "$stage"/* "$stage"/.[!.]* "$stage"/..?*; do
        [ -e "$entry" ] || continue
        mv "$entry" "$dir"/
    done
    rmdir "$stage"
    chmod 755 "$dir"
}

# internal_verify ARCHIVE: restore the dump into a private PostgreSQL server inside this
# container and compare it with the manifest. The live deployment is never contacted.
internal_verify() {
    local archive=$1 tmp root sock pg_bin status=0
    tmp=$(mktemp -d)
    CLEANUP_PATHS+=("$tmp")
    info "Checking archive checksums..."
    internal_validate "$archive" >"$tmp/manifest.json"

    pg_bin=$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -n 1)
    [ -n "$pg_bin" ] || die "PostgreSQL server binaries are not available in this image."
    root=$(mktemp -d /tmp/morphik-verify.XXXXXX)
    CLEANUP_PATHS+=("$root")
    sock="$root/socket"
    mkdir -p "$root/data" "$sock"
    chown -R postgres:postgres "$root"
    chmod 700 "$root"

    info "Starting a temporary PostgreSQL server..."
    gosu postgres "$pg_bin/initdb" -D "$root/data" -U "$PG_USER" --auth=trust >"$root/initdb.log" 2>&1 || {
        cat "$root/initdb.log" >&2
        die "initdb failed."
    }
    gosu postgres "$pg_bin/pg_ctl" -D "$root/data" -l "$root/server.log" -w \
        -o "-c listen_addresses='' -k $sock -c fsync=off -c full_page_writes=off -c synchronous_commit=off -c maintenance_work_mem=256MB" \
        start >/dev/null || {
        cat "$root/server.log" >&2
        die "The temporary PostgreSQL server did not start."
    }
    export PGHOST="$sock"

    createdb -U "$PG_USER" "$PG_DB"
    info "Restoring the database dump..."
    if ! tar -xOf "$archive" --occurrence=1 database.dump |
        pg_restore -U "$PG_USER" -d "$PG_DB" --no-owner --single-transaction --exit-on-error 2>"$tmp/restore.err"; then
        cat "$tmp/restore.err" >&2
        gosu postgres "$pg_bin/pg_ctl" -D "$root/data" -m fast stop >/dev/null 2>&1 || true
        die "pg_restore failed."
    fi

    info "Comparing row counts and checksums with the manifest..."
    database_stats >"$tmp/stats.json"
    json_tool compare "$tmp/manifest.json" "$tmp/stats.json" || status=1

    if [ "$(json_tool get "$tmp/manifest.json" storage.included)" = "true" ]; then
        info "Checking that every referenced source file is in the archive..."
        if psql -X -At -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" -c "SELECT to_regclass('public.documents') IS NOT NULL" | grep -qx t; then
            psql -X -At -q -F $'\t' -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" -c "$REFERENCED_FILES_SQL" >"$tmp/refs.tsv"
        else
            : >"$tmp/refs.tsv"
        fi
        tar -xOf "$archive" --occurrence=1 storage.tar | tar -tf - >"$tmp/listing"
        if ! json_tool missing-files "$tmp/refs.tsv" "$tmp/listing" "$(json_tool get "$tmp/manifest.json" storage.storage_path)" >"$tmp/missing"; then
            status=1
            printf 'MISMATCH: %s referenced source files are missing from storage.tar:\n' "$(wc -l <"$tmp/missing" | tr -d ' ')" >&2
            head -n 20 "$tmp/missing" | sed 's/^/  /' >&2
        fi
    fi

    gosu postgres "$pg_bin/pg_ctl" -D "$root/data" -m fast stop >/dev/null 2>&1 || true
    json_tool summary "$tmp/manifest.json" | sed 's/^/  /' >&2
    if [ "$status" -ne 0 ]; then
        die "Verification failed."
    fi
    info "PASS: the restored database matches the manifest."
}

# internal_retention DIR KEEP: delete the oldest scheduled backups beyond KEEP. Only files named
# morphik-<timestamp>-auto.backup are candidates, so manual and pre-restore backups are kept.
internal_retention() {
    local dir=$1 keep=$2 file count=0
    case "$keep" in
        '' | *[!0-9]*) die "keep must be a positive integer." ;;
    esac
    [ "$keep" -ge 1 ] || die "keep must be at least 1."
    for file in $(auto_backups "$dir" | sort -r); do
        count=$((count + 1))
        if [ "$count" -gt "$keep" ]; then
            rm -f "$dir/$file"
            info "Retention removed $file."
        fi
    done
}

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

backup_dir() {
    local dir=${1:-}
    if [ -z "$dir" ]; then
        if [ "$IN_CONTAINER" = "1" ]; then
            dir=/backups
        else
            dir=$(toml_get "$CONFIG_FILE" backup directory)
            dir=${dir:-./backups}
        fi
    fi
    case "$dir" in
        /*) ;;
        *) dir="$INSTALL_DIR/${dir#./}" ;;
    esac
    printf '%s\n' "$dir"
}

morphik_version() {
    local version=${MORPHIK_VERSION:-}
    if [ -z "$version" ] && [ -f .env ]; then
        version=$(sed -n 's/^MORPHIK_VERSION=//p' .env | tail -n 1)
    fi
    printf '%s\n' "${version:-latest}"
}

core_facts() {
    local ref="" image_id="" cid digests="" revision=""
    if [ "$IN_CONTAINER" = "0" ]; then
        cid=$(compose ps -q morphik 2>/dev/null | head -n 1 || true)
        if [ -n "$cid" ]; then
            ref=$(docker inspect -f '{{.Config.Image}}' "$cid" 2>/dev/null || true)
            image_id=$(docker inspect -f '{{.Image}}' "$cid" 2>/dev/null || true)
        fi
        [ -n "$ref" ] || ref="ghcr.io/morphik-org/morphik-core:$(morphik_version)"
        [ -n "$image_id" ] || image_id=$(docker image inspect -f '{{.Id}}' "$ref" 2>/dev/null || true)
        if [ -n "$image_id" ]; then
            digests=$(docker image inspect -f '{{json .RepoDigests}}' "$image_id" 2>/dev/null || true)
            revision=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image_id" 2>/dev/null || true)
            [ "$revision" = "<no value>" ] && revision=""
        fi
    else
        ref="ghcr.io/morphik-org/morphik-core:${MORPHIK_VERSION:-latest}"
    fi
    fact s core.image "$ref"
    fact s core.image_id "$image_id"
    fact j core.repo_digests "${digests:-null}"
    fact s core.revision "$revision"
}

# create_backup OUT_DIR SUFFIX MODE INCLUDE_ENV. Sets CREATED_BACKUP to the archive path.
create_backup() {
    local out_dir=$1 suffix=$2 mode=$3 include_env=$4
    local stamp created_at name final partial work storage_provider
    mkdir -p "$out_dir"
    chmod 700 "$out_dir" 2>/dev/null || true
    [ -f "$CONFIG_FILE" ] || die "morphik.toml not found at $CONFIG_FILE."

    stamp=$(utc_stamp)
    while [ -e "$out_dir/morphik-${stamp}${suffix}.backup" ]; do
        sleep 1
        stamp=$(utc_stamp)
    done
    created_at=$(utc_iso)
    name="morphik-${stamp}${suffix}.backup"
    final="$out_dir/$name"
    partial="$out_dir/.$name.partial"
    work=$(mktemp -d "$out_dir/.work-${stamp}.XXXXXX")
    CLEANUP_PATHS+=("$work" "$partial")
    mkdir "$work/config"

    info "Dumping PostgreSQL (documents, metadata, folders, chunks, embeddings)..."
    dump_database "$work/database.dump" "$work/database.json"

    storage_provider=$(toml_get "$CONFIG_FILE" storage provider)
    storage_provider=${storage_provider:-local}
    if [ "$storage_provider" = "local" ]; then
        info "Copying source files from $STORAGE_DIR..."
        if [ "$IN_CONTAINER" = "1" ]; then
            internal_storage_tar "$STORAGE_DIR" >"$work/storage.tar"
        else
            [ -d "$STORAGE_DIR" ] || die "Storage directory $STORAGE_DIR does not exist."
            run_tool -v "$(abs_path "$STORAGE_DIR"):/storage:ro" -- _storage-tar /storage >"$work/storage.tar"
        fi
    else
        warn "Storage provider is '$storage_provider'. Source files stay in that bucket and are NOT in this backup."
        warn "Protect the bucket separately, for example with S3 versioning."
    fi

    cat "$CONFIG_FILE" >"$work/config/morphik.toml"
    if [ "$include_env" = "1" ]; then
        [ -f "$ENV_FILE" ] || die "--include-env was given, but $ENV_FILE does not exist."
        cat "$ENV_FILE" >"$work/config/.env"
        warn "This backup includes .env. It contains secrets. Store it like a password."
    fi

    {
        fact s created_at "$created_at"
        fact s name "$name"
        fact s created_by.tool "morphik-backup.sh"
        fact s created_by.mode "$mode"
        fact b includes_env "$([ "$include_env" = "1" ] && echo true || echo false)"
        core_facts
        config_facts "$CONFIG_FILE"
        if [ "$storage_provider" != "local" ]; then
            fact s storage.note "Source files are in the '$storage_provider' storage provider and were not copied."
        fi
    } >"$work/facts.tsv"

    info "Writing $name..."
    if [ "$IN_CONTAINER" = "1" ]; then
        internal_package "$work" >"$partial"
    else
        run_tool -v "$work:/work:ro" -- _package /work >"$partial"
    fi
    mv "$partial" "$final"
    chmod 600 "$final"
    if [ "$IN_CONTAINER" = "1" ] && [ -n "${MORPHIK_BACKUP_OWNER:-}" ]; then
        chown "$MORPHIK_BACKUP_OWNER" "$final" 2>/dev/null || warn "Could not change the owner of $final."
    fi
    rm -rf "$work"
    CREATED_BACKUP=$final
}

AWS_VARIABLES="AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION AWS_DEFAULT_REGION AWS_PROFILE AWS_ENDPOINT_URL"

# Credentials come from the shell, then .env (as for the backup-s3 service), then the EC2
# instance role. Values go through an owner-only env file, never the command line.
upload_to_s3() {
    local file=$1 uri=$2 region=$3 name env_args=() var aws_env
    name=$(basename "$file")
    aws_env=$(mktemp "${TMPDIR:-/tmp}/morphik-backup-aws.XXXXXX")
    CLEANUP_PATHS+=("$aws_env")
    if [ -f "$ENV_FILE" ]; then
        # docker --env-file keeps quotes that Compose would strip, so strip them here.
        { grep -E "^($(printf '%s' "$AWS_VARIABLES" | tr ' ' '|'))=" "$ENV_FILE" || true; } |
            sed -e 's/^\([A-Z_]*\)="\(.*\)"$/\1=\2/' -e "s/^\([A-Z_]*\)='\(.*\)'$/\1=\2/" >"$aws_env"
    fi
    env_args+=(--env-file "$aws_env")
    for var in $AWS_VARIABLES; do
        if [ -n "${!var:-}" ]; then
            env_args+=(-e "$var")
        fi
    done
    if [ -n "$region" ]; then
        env_args+=(-e "AWS_DEFAULT_REGION=$region")
    fi
    if [ -d "${HOME:-}/.aws" ]; then
        env_args+=(-v "$HOME/.aws:/root/.aws:ro")
    fi
    # Only needed when the S3 endpoint is itself a container, such as an on-prem MinIO.
    if [ -n "${MORPHIK_BACKUP_S3_NETWORK:-}" ]; then
        env_args+=(--network "$MORPHIK_BACKUP_S3_NETWORK")
    fi
    info "Uploading $name to ${uri%/}/..."
    docker run --rm ${env_args[@]+"${env_args[@]}"} -v "$file:/upload/$name:ro" \
        "$AWS_IMAGE" s3 cp "/upload/$name" "${uri%/}/$name" --only-show-errors --no-progress
}

cmd_backup() {
    local include_env=0 output_dir="" upload=1 file s3_uri s3_region
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --include-env) include_env=1 ;;
            --output-dir)
                [ "$#" -ge 2 ] || die "--output-dir needs a directory."
                output_dir=$2
                shift
                ;;
            --output-dir=*) output_dir=${1#*=} ;;
            --no-upload) upload=0 ;;
            -h | --help) usage 0 ;;
            *) die "Unknown backup option: $1" ;;
        esac
        shift
    done
    require_host_deployment
    require_running_postgres
    TOOL_IMAGE=$(tool_image)

    create_backup "$(backup_dir "$output_dir")" "" manual "$include_env"
    file=$CREATED_BACKUP
    info ""
    info "Backup written: $file"
    run_tool -v "$file:$ARCHIVE_IN_TOOL:ro" -- _summary "$ARCHIVE_IN_TOOL" | sed 's/^/  /' >&2

    s3_uri=$(toml_get "$CONFIG_FILE" backup s3_uri)
    s3_region=$(toml_get "$CONFIG_FILE" backup s3_region)
    if [ "$upload" = "1" ] && [ -n "$s3_uri" ]; then
        if ! upload_to_s3 "$file" "$s3_uri" "$s3_region"; then
            die "The backup was written locally, but the upload to $s3_uri failed."
        fi
        info "Uploaded to ${s3_uri%/}/$(basename "$file")"
    elif [ -z "$s3_uri" ]; then
        info "This backup is on the same disk as Morphik. Copy it to another machine or set [backup] s3_uri."
    fi
    printf '%s\n' "$file"
}

internal_summary() {
    local tmp
    tmp=$(mktemp -d)
    CLEANUP_PATHS+=("$tmp")
    tar -xOf "$1" --occurrence=1 manifest.json >"$tmp/manifest.json"
    json_tool summary "$tmp/manifest.json"
    local documents
    documents=$(json_tool get "$tmp/manifest.json" counts.documents)
    if [ "$documents" = "0" ]; then
        printf '%s\n' "WARNING: the database had no documents. If you expected data, check COMPOSE_PROJECT_NAME and the install directory name: a changed project name points Compose at a new, empty volume."
    fi
}

# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

restore_failure_note() {
    case "$RESTORE_STAGE" in
        services-stopped)
            warn "Restore stopped before changing any data. Start Morphik again with ./start-morphik.sh."
            ;;
        database)
            warn "Restore failed while the database was being replaced. The morphik and worker services are still stopped."
            warn "Fix the problem and run the restore again. ${SAFETY_BACKUP:+Your previous data is in $SAFETY_BACKUP.}"
            ;;
        storage | config)
            warn "Restore failed after the database was restored. The morphik and worker services are still stopped."
            warn "Run the restore again. ${SAFETY_BACKUP:+Your previous data is in $SAFETY_BACKUP.}"
            ;;
    esac
}

api_port() {
    local port
    port=$(toml_get "$CONFIG_FILE" api port)
    printf '%s\n' "${port:-8000}"
}

target_data() {
    pg psql -X -At -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" -c "$TARGET_DATA_SQL"
}

storage_has_files() {
    [ -d "$STORAGE_DIR" ] || return 1
    [ -n "$(find "$STORAGE_DIR" -mindepth 1 -maxdepth 1 ! -name '.morphik-restore-*' 2>/dev/null | head -n 1)" ]
}

services_defined() {
    compose --profile "*" config --services 2>/dev/null
}

stop_app_services() {
    local services=(morphik worker) defined service
    defined=$(services_defined)
    for service in backup backup-s3; do
        if printf '%s\n' "$defined" | grep -qx "$service"; then
            services+=("$service")
        fi
    done
    info "Stopping ${services[*]}..."
    compose --profile "*" stop "${services[@]}" >/dev/null
}

start_app_services() {
    if [ -x ./start-morphik.sh ]; then
        ./start-morphik.sh
    else
        compose up -d morphik worker
    fi
}

cmd_restore() {
    local file="" force=0 restore_config=0 safety=1 start=1
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --force) force=1 ;;
            --restore-config) restore_config=1 ;;
            --no-safety-backup) safety=0 ;;
            --no-start) start=0 ;;
            -h | --help) usage 0 ;;
            -*) die "Unknown restore option: $1" ;;
            *)
                [ -z "$file" ] || die "restore takes one backup file."
                file=$1
                ;;
        esac
        shift
    done
    [ -n "$file" ] || die "Usage: $SCRIPT_NAME restore FILE [--force]"
    [ -f "$file" ] || die "Backup file not found: $file"
    file=$(abs_path "$file")
    require_host_deployment

    local bdir work stamp existing="" project volume stuck_count name
    bdir=$(backup_dir "")
    mkdir -p "$bdir"
    stamp=$(utc_stamp)
    work=$(mktemp -d "$bdir/.restore-${stamp}.XXXXXX")
    CLEANUP_PATHS+=("$work")
    TOOL_IMAGE=$(tool_image)
    local archive_mount=(-v "$file:$ARCHIVE_IN_TOOL:ro")

    info "Checking $(basename "$file")..."
    run_tool "${archive_mount[@]}" -- _validate "$ARCHIVE_IN_TOOL" >"$work/manifest.json" ||
        die "The backup did not pass validation. Nothing was changed."
    run_tool -v "$work:/work:ro" -- _json summary /work/manifest.json | sed 's/^/  /' >&2
    name=$(run_tool -v "$work:/work:ro" -- _json get /work/manifest.json name)

    if [ "$restore_config" = "1" ]; then
        run_tool "${archive_mount[@]}" -- _extract "$ARCHIVE_IN_TOOL" config/morphik.toml >"$work/target.toml" ||
            die "The backup has no config/morphik.toml."
        if [ "$(run_tool -v "$work:/work:ro" -- _json get /work/manifest.json includes_env)" = "true" ]; then
            run_tool "${archive_mount[@]}" -- _extract "$ARCHIVE_IN_TOOL" config/.env >"$work/target.env"
        fi
    else
        [ -f "$CONFIG_FILE" ] || die "morphik.toml not found. Use --restore-config to take it from the backup."
        cat "$CONFIG_FILE" >"$work/target.toml"
    fi

    info "Checking the backup against the target configuration..."
    if ! run_tool -v "$work:/work:ro" -- _compat /work/manifest.json /work/target.toml >&2; then
        die "The backup does not match this deployment's morphik.toml. Nothing was changed.
Restoring would leave embeddings that the configured model cannot search. Either edit morphik.toml to
match the backup, or re-run with --restore-config to use the configuration saved in the backup."
    fi

    info "Starting PostgreSQL..."
    local up_output
    up_output=$(compose up -d postgres 2>&1) || {
        printf '%s\n' "$up_output" >&2
        die "Could not start PostgreSQL."
    }
    wait_for_postgres
    project=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$(postgres_container)")
    volume=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$(postgres_container)")
    info "Target: Compose project '$project', PostgreSQL volume '$volume', storage $STORAGE_DIR"

    check_target_empty() {
        existing=$(target_data)
        if storage_has_files; then
            existing="${existing:+$existing }storage-files=yes"
        fi
        [ -z "$existing" ]
    }
    if ! check_target_empty && [ "$force" = "0" ]; then
        die "The target already has data: $existing
Nothing was changed. Re-run with --force to replace it. With --force, the script first writes a
safety backup of the current data unless you pass --no-safety-backup."
    fi

    stop_app_services
    RESTORE_STAGE=services-stopped
    if ! check_target_empty; then
        if [ "$force" = "0" ]; then
            RESTORE_STAGE=""
            start_app_services >/dev/null 2>&1 || true
            die "Data was written while the services were stopping: $existing. Nothing was changed."
        fi
        if [ "$safety" = "1" ]; then
            info "Writing a safety backup of the current data first..."
            create_backup "$bdir" "-pre-restore" pre-restore 0
            SAFETY_BACKUP=$CREATED_BACKUP
            info "Safety backup: $SAFETY_BACKUP"
        else
            warn "Replacing existing data without a safety backup (--no-safety-backup)."
        fi
    fi

    RESTORE_STAGE=database
    info "Replacing the database..."
    pg psql -X -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d postgres \
        -c "DROP DATABASE IF EXISTS \"$PG_DB\" WITH (FORCE)" \
        -c "CREATE DATABASE \"$PG_DB\"" >/dev/null
    run_tool "${archive_mount[@]}" -- _extract "$ARCHIVE_IN_TOOL" database.dump |
        pg pg_restore -U "$PG_USER" -d "$PG_DB" --clean --if-exists --no-owner --single-transaction --exit-on-error

    info "Comparing the restored database with the manifest..."
    database_stats pg >"$work/stats.json"
    run_tool -v "$work:/work:ro" -- _json compare /work/manifest.json /work/stats.json >&2 ||
        die "The restored database does not match the backup manifest."

    : >"$work/stuck"
    if [ "$(pg psql -X -At -q -U "$PG_USER" -d "$PG_DB" -c "SELECT to_regclass('public.documents') IS NOT NULL")" = "t" ]; then
        pg psql -X -At -q -v ON_ERROR_STOP=1 -v backup_name="$name" -U "$PG_USER" -d "$PG_DB" \
            <<<"$STUCK_DOCUMENTS_SQL" >"$work/stuck"
    fi
    stuck_count=$(grep -c . "$work/stuck" || true)

    RESTORE_STAGE=storage
    if [ "$(run_tool -v "$work:/work:ro" -- _json get /work/manifest.json storage.included)" = "true" ]; then
        info "Replacing source files in $STORAGE_DIR..."
        mkdir -p "$STORAGE_DIR"
        run_tool "${archive_mount[@]}" -v "$(abs_path "$STORAGE_DIR"):/storage" -- _restore-storage "$ARCHIVE_IN_TOOL" /storage
    else
        warn "The backup has no source files (storage was not local). They must still exist in the storage bucket."
    fi

    RESTORE_STAGE=config
    if [ "$restore_config" = "1" ]; then
        if [ -f "$CONFIG_FILE" ]; then
            cp -p "$CONFIG_FILE" "$CONFIG_FILE.before-restore-$stamp"
            info "Saved the previous morphik.toml as morphik.toml.before-restore-$stamp"
        fi
        cat "$work/target.toml" >"$CONFIG_FILE"
        if [ -f "$work/target.env" ]; then
            if [ -f "$ENV_FILE" ]; then
                cp -p "$ENV_FILE" "$ENV_FILE.before-restore-$stamp"
                info "Saved the previous .env as $(basename "$ENV_FILE").before-restore-$stamp"
            fi
            cat "$work/target.env" >"$ENV_FILE"
            chmod 600 "$ENV_FILE"
        fi
    fi
    RESTORE_STAGE=""

    info ""
    info "Restore complete."
    if [ "$stuck_count" -gt 0 ]; then
        local requeue_file="$bdir/restore-$stamp-requeue.json"
        cp "$work/stuck" "$work/stuck.ids"
        run_tool -v "$work:/work:ro" -- _json requeue-body /work/stuck.ids >"$requeue_file"
        info "$stuck_count document(s) were still ingesting when the backup was taken."
        info "They are now marked failed, with a new ingestion revision so no stale job can overwrite them."
        info "After Morphik starts, requeue them:"
        info "  curl -X POST http://localhost:$(api_port)/ingest/requeue \\"
        info "    -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \\"
        info "    -d @$requeue_file"
    fi
    if [ "$start" = "1" ]; then
        info "Starting Morphik..."
        start_app_services
    else
        info "Services were left stopped (--no-start). Start them with ./start-morphik.sh."
    fi
}

# ---------------------------------------------------------------------------
# Verify and list
# ---------------------------------------------------------------------------

cmd_verify() {
    local file=${1:-}
    [ -n "$file" ] || die "Usage: $SCRIPT_NAME verify FILE"
    [ -f "$file" ] || die "Backup file not found: $file"
    file=$(abs_path "$file")
    if [ "$IN_CONTAINER" = "1" ]; then
        internal_verify "$file"
        return
    fi
    command -v docker >/dev/null 2>&1 || die "Docker is required."
    TOOL_IMAGE=${MORPHIK_BACKUP_TOOL_IMAGE:-$DEFAULT_TOOL_IMAGE}
    if [ -f "$COMPOSE_FILE" ]; then
        TOOL_IMAGE=$(tool_image)
    fi
    info "Verifying $(basename "$file") in a temporary PostgreSQL container. The live deployment is not touched."
    run_tool -v "$file:$ARCHIVE_IN_TOOL:ro" -- _verify "$ARCHIVE_IN_TOOL"
}

cmd_list() {
    local output_dir=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --output-dir)
                output_dir=$2
                shift
                ;;
            --output-dir=*) output_dir=${1#*=} ;;
            *) die "Unknown list option: $1" ;;
        esac
        shift
    done
    local dir
    dir=$(backup_dir "$output_dir")
    [ -d "$dir" ] || {
        info "No backups in $dir."
        return 0
    }
    (cd "$dir" && { ls -1 2>/dev/null | grep -E '^morphik-.*\.backup$' || true; } | sort | while IFS= read -r file; do
        ls -lh "$file" | awk '{ printf "%-50s %8s  %s %s %s\n", $NF, $5, $6, $7, $8 }'
    done)
}

# ---------------------------------------------------------------------------
# Scheduled backups (backup service) and off-host copies (backup-s3 service)
# ---------------------------------------------------------------------------

backup_setting() {
    toml_get "$CONFIG_FILE" backup "$1"
}

auto_backups() {
    ls -1 "$1" 2>/dev/null | grep -E '^morphik-[0-9]{8}T[0-9]{6}Z-auto\.backup$' | sort || true
}

cmd_scheduled() {
    [ "$IN_CONTAINER" = "1" ] || die "scheduled runs inside the backup service. Enable it with [backup] enabled = true in morphik.toml."
    local dir=/backups enabled interval_hours interval keep verify include_env newest age now file wait_s
    mkdir -p "$dir"
    info "Scheduled backups started. Reading [backup] from morphik.toml before each run."
    while true; do
        enabled=$(backup_setting enabled)
        if [ "$enabled" != "true" ]; then
            info "[backup] enabled is not true. Checking again in 5 minutes."
            pause 300
            continue
        fi
        interval_hours=$(backup_setting interval_hours)
        interval=$(awk -v h="${interval_hours:-24}" -v min="${MORPHIK_BACKUP_MIN_INTERVAL_SECONDS:-60}" \
            'BEGIN { s = int(h * 3600); if (s < min) s = min; print s }')
        keep=$(backup_setting keep)
        keep=${keep:-7}
        verify=$(backup_setting verify)
        include_env=0
        [ "$(backup_setting include_env)" = "true" ] && include_env=1

        newest=$(auto_backups "$dir" | tail -n 1)
        now=$(date +%s)
        age=$interval
        if [ -n "$newest" ]; then
            age=$((now - $(stat -c %Y "$dir/$newest")))
        fi
        if [ "$age" -lt "$interval" ]; then
            wait_s=$((interval - age))
            [ "$wait_s" -gt 300 ] && wait_s=300
            pause "$wait_s"
            continue
        fi

        info "$(utc_iso) starting scheduled backup."
        # A subshell keeps one failed run from stopping the loop. It needs its own cleanup trap.
        if file=$(
            trap cleanup EXIT
            create_backup "$dir" "-auto" scheduled "$include_env"
            printf '%s' "$CREATED_BACKUP"
        ); then
            info "$(utc_iso) wrote $(basename "$file")."
            if [ "$verify" = "true" ]; then
                if (
                    trap cleanup EXIT
                    internal_verify "$file"
                ); then
                    info "$(utc_iso) verified $(basename "$file")."
                else
                    warn "$(utc_iso) verification failed for $(basename "$file"). Keeping older backups."
                    pause 900
                    continue
                fi
            fi
            (internal_retention "$dir" "$keep") || warn "Retention failed."
        else
            warn "$(utc_iso) scheduled backup failed. Retrying in 15 minutes."
            pause 900
        fi
    done
}

cmd_offsite_sync() {
    [ "$IN_CONTAINER" = "1" ] || die "offsite-sync runs inside the backup-s3 service."
    local uri region interval=${MORPHIK_BACKUP_SYNC_SECONDS:-300}
    info "Off-host copy started. Uploads new backups from /backups every $interval seconds."
    while true; do
        uri=$(backup_setting s3_uri)
        region=$(backup_setting s3_region)
        if [ -z "$uri" ]; then
            info "[backup] s3_uri is empty. Checking again in 5 minutes."
            pause 300
            continue
        fi
        if [ -n "$region" ]; then
            export AWS_DEFAULT_REGION="$region"
        fi
        # sync never deletes remote objects. Use an S3 lifecycle rule for remote retention.
        if aws s3 sync /backups "${uri%/}/" --exclude '*' --include 'morphik-*.backup' --only-show-errors --no-progress; then
            info "$(utc_iso) backups are copied to $uri."
        else
            warn "$(utc_iso) copy to $uri failed. Retrying in $interval seconds."
        fi
        pause "$interval"
    done
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage:
  $SCRIPT_NAME backup [--include-env] [--output-dir DIR] [--no-upload]
      Write one backup file: PostgreSQL (with embeddings), source files, morphik.toml, and a manifest.
      --include-env   also save .env. It holds secrets, so it is left out by default.
      --no-upload     skip the S3 copy even when [backup] s3_uri is set.

  $SCRIPT_NAME restore FILE [--force] [--restore-config] [--no-safety-backup] [--no-start]
      Check the file, stop morphik and worker, replace the database and source files, start again.
      --force             replace a deployment that already has data. A safety backup is written first.
      --restore-config    also restore morphik.toml (and .env, if the backup has it).
      --no-safety-backup  with --force, skip the safety backup.
      --no-start          leave the services stopped afterwards.

  $SCRIPT_NAME verify FILE
      Restore the file into a temporary PostgreSQL container and compare it with the manifest.

  $SCRIPT_NAME list [--output-dir DIR]
      List backup files.
EOF
    exit "${1:-0}"
}

main() {
    local command=${1:-}
    [ "$#" -gt 0 ] && shift
    if [ "${MORPHIK_BACKUP_TOOL:-}" != "1" ] && [ "$IN_CONTAINER" = "0" ]; then
        cd "$INSTALL_DIR"
    fi
    case "$command" in
        backup) cmd_backup "$@" ;;
        restore) cmd_restore "$@" ;;
        verify) cmd_verify "$@" ;;
        list) cmd_list "$@" ;;
        scheduled) cmd_scheduled "$@" ;;
        offsite-sync) cmd_offsite_sync "$@" ;;
        _storage-tar) internal_storage_tar "$@" ;;
        _package) internal_package "$@" ;;
        _validate) internal_validate "$@" ;;
        _extract) internal_extract "$@" ;;
        _compat) internal_compat "$@" ;;
        _config-facts) config_facts "$@" ;;
        _manifest) internal_manifest "$@" ;;
        _restore-storage) internal_restore_storage "$@" ;;
        _verify) internal_verify "$@" ;;
        _retention) internal_retention "$@" ;;
        _summary) internal_summary "$@" ;;
        _json) json_tool "$@" ;;
        -h | --help | help | "") usage 0 ;;
        *)
            info "Unknown command: $command"
            usage 2
            ;;
    esac
}

main "$@"
