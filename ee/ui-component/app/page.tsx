"use client";

import React, { Suspense } from 'react';
import MorphikUI from '@/components/MorphikUI';
import { useSearchParams } from 'next/navigation';

type AllowedSection = 'documents' | 'search' | 'chat' | 'graphs' | 'agent';
const ALLOWED_SECTIONS: AllowedSection[] = ['documents', 'search', 'chat', 'graphs', 'agent'];

function isValidSection(section: string | null): section is AllowedSection {
  return section !== null && ALLOWED_SECTIONS.includes(section as AllowedSection);
}

function HomeContent() {
  const searchParams = useSearchParams();
  const folderParam = searchParams.get('folder');
  const sectionParam = searchParams.get('section');

  const initialSectionValidated: AllowedSection | undefined = isValidSection(sectionParam)
    ? sectionParam
    : undefined;

  return <MorphikUI initialFolder={folderParam} initialSection={initialSectionValidated} />;
}

export default function Home() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HomeContent />
    </Suspense>
  );
}
