#!/usr/bin/env python3
import re
import sys
import urllib.parse
from typing import Dict

def parse_postgres_uri(uri: str) -> Dict[str, str]:
    """Parse a PostgreSQL connection URI into its components.
    
    Args:
        uri: The PostgreSQL connection URI (e.g., 'postgresql://user:pass@host:port/db')
        
    Returns:
        Dictionary containing the parsed components (USER, PASS, HOST, PORT, DB)
    """
    # Default values
    result = {
        'USER': '',
        'PASS': '',
        'HOST': '',
        'PORT': '5432',  # Default PostgreSQL port
        'DB': 'postgres'  # Default database name
    }
    
    try:
        # PostgreSQL URI pattern:
        # postgresql[+driver]://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
        pattern = r'''
            ^
            postgresql(?:\+[a-z]+)?://  # scheme with optional driver
            (?:([^:/?#@]+)(?::([^/?#@]*))?@)?  # user:password@
            (?:([^:/?#]+)(?::(\d+))?)?  # host:port
            (?:/([^?#]*))?  # /dbname
            (?:\?([^#]*))?  # ?query
            $
        '''
        
        match = re.match(pattern, uri.strip(), re.VERBOSE)
        if not match:
            raise ValueError("Invalid PostgreSQL URI format")
            
        user, password, host, port, dbname, query = match.groups()
        
        # Handle username and password
        if user:
            result['USER'] = urllib.parse.unquote(user)
        if password:
            result['PASS'] = urllib.parse.unquote(password)
        
        # Handle host and port
        if host:
            result['HOST'] = host
        if port:
            result['PORT'] = port
        
        # Handle database name
        if dbname:
            result['DB'] = urllib.parse.unquote(dbname)
        
        # Handle query parameters (e.g., for password in query string)
        if query and not result['PASS']:
            for param in query.split('&'):  # type: ignore
                if '=' in param:
                    key, value = param.split('=', 1)
                    if key.lower() == 'password':
                        result['PASS'] = urllib.parse.unquote(value)
                        break
    
    except Exception as e:
        # If any error occurs, print the failure message and exit
        print(f"Error parsing PostgreSQL URI: {e}", file=sys.stderr)
        print("POSTGRES_URI_PARSE_FAILURE=1", file=sys.stderr)
        sys.exit(1)
    
    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: eval $(parse-postgres-uri.sh \"postgresql://user:pass@host:port/db\")", file=sys.stderr)
        sys.exit(1)
    
    # Print default empty values first (for compatibility with original script)
    print('PG_USER=""')
    print('PG_PASS=""')
    print('PG_HOST=""')
    print('PG_PORT=""')
    print('PG_DB=""')
    
    # Parse the URI and print the results
    try:
        components = parse_postgres_uri(sys.argv[1])
        for key, value in components.items():
            # Escape special characters in the value for shell compatibility
            escaped_value = value.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
            print(f'PG_{key}="{escaped_value}"')
    except Exception as e:
        print(f'Error parsing PostgreSQL URI: {e}', file=sys.stderr)
        print('POSTGRES_URI_PARSE_FAILURE=1')
        sys.exit(1)

if __name__ == "__main__":
    main()