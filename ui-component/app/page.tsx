"use client";

import React from 'react';
import MorphikUI from '@/components/MorphikUI';
import { useSearchParams } from 'next/navigation';

export default function Home() {
  const searchParams = useSearchParams();
  const folderParam = searchParams.get('folder');
  
  return <MorphikUI initialFolder={folderParam} />;
}