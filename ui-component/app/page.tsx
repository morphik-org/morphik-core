"use client";

import React, { Suspense } from 'react';
import MorphikUI from '@/components/MorphikUI';
import { useSearchParams } from 'next/navigation';

function HomeContent() {
  const searchParams = useSearchParams();
  const folderParam = searchParams.get('folder');
  
  return <MorphikUI initialFolder={folderParam} />;
}

export default function Home() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HomeContent />
    </Suspense>
  );
}