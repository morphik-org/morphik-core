# Design Guide

This document contains the design aesthetic and code examples from our landing page that should be used as the foundation for our UI component design system.

Note that your job is to copy the design aesthetic of the landing page, NOT to actually change the functionality and stuff of the system. for example, the buttons should look consistent, the typography should be consistent, and the general look and feel should be consistent.

the goal isn't to stick a new landing apge into the system, it is to actually have a really nice looking UI, and I've focus on distilling most of our key priciples into that landing page.

If you feel like getting more of a feel of it, you can use the pupeteer MCP and navigate to https://morphik.ai and that should show you the entire landing page.

Here's the code for the landing page:

```tsx
import SpotlightComponent from '@/components/spotlight';
import BenchmarkShowcase from '@/components/benchmark-showcase';
import { BentoGridThirdDemo } from '@/components/bento-grid-demo';
import WhyMorphikSection from '@/components/why-morphik-section';
import Navbar from '@/components/navbar';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-white">
      <Navbar />

      {/* Hero Section */}
      <section className="pb-6 pt-32 md:pb-16 md:pt-56">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-start md:items-center">
            <div className="mb-6 text-left md:text-center">
              <h1>
                <span className="font-bodoni text-4xl font-thin leading-tight text-black md:text-4xl lg:text-5xl">
                  Build Agents that{' '}
                  <em className="font-bodoni font-thin italic">Never</em>{' '}
                  Hallucinate
                </span>
              </h1>
            </div>

            <div className="mb-4 text-left md:mb-8 md:text-center">
              <p className="max-w-2xl font-sans text-base text-gray-600 md:px-0 md:text-lg lg:text-xl">
                Deploy the most accurate RAG in the world in two lines of code
              </p>
            </div>

            {/* Y Combinator section - mobile only, positioned after subtitle */}
            <a
              href="https://www.ycombinator.com/companies/morphik"
              target="_blank"
              rel="noopener noreferrer"
              className="mb-8 flex items-start justify-start transition-opacity hover:opacity-80 md:hidden"
            >
              <span className="text-sm font-medium text-gray-600">
                Backed By Y Combinator
              </span>
            </a>

            <div className="mb-8 flex w-full flex-col space-y-3 sm:flex-row sm:justify-center sm:space-x-4 sm:space-y-0">
              <a
                href="/signup"
                className="w-full rounded-md bg-black px-6 py-3 text-center font-sans text-white transition-colors hover:bg-gray-800 sm:w-auto"
              >
                Get Started
              </a>
              <a
                href="https://docs.morphik.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full rounded-md border border-black px-6 py-3 text-center font-sans text-black transition-colors hover:bg-gray-50 sm:w-auto"
              >
                Documentation
              </a>
            </div>

            {/* Y Combinator Backed By Section - desktop only */}
            <a
              href="https://www.ycombinator.com/companies/morphik"
              target="_blank"
              rel="noopener noreferrer"
              className="mb-12 hidden items-center justify-center transition-opacity hover:opacity-80 md:flex"
            >
              <span className="text-sm font-medium text-gray-600">
                Backed By Y Combinator
              </span>
            </a>

            <div className="hidden w-full overflow-hidden rounded-lg md:block">
              <SpotlightComponent />
            </div>
          </div>
        </div>
      </section>

      {/* Mobile Spotlight Section - Below the fold on mobile only */}
      <section className="block pb-12 pt-6 md:hidden">
        <div className="container mx-auto px-4">
          <div className="w-full overflow-hidden rounded-lg">
            <SpotlightComponent />
          </div>
        </div>
      </section>

      {/* Customer Logos Section */}

      {/* Benchmark Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center">
            <div className="mb-16 w-full">
              <div className="mb-8 text-left">
                <h2 className="mb-4 font-bodoni text-3xl font-thin text-black md:text-4xl">
                  State-of-the-Art Performance
                </h2>
                <p className="font-sans text-lg text-gray-600">
                  Morphik consistently outperforms traditional RAG systems and
                  leading LLMs on challenging document analysis benchmarks
                </p>
              </div>
              <BenchmarkShowcase />
            </div>
          </div>
        </div>
      </section>

      {/* Why Morphik Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center">
            <div className="w-full">
              <WhyMorphikSection />
            </div>
          </div>
        </div>
      </section>

      {/* Bento Grid Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center">
            <div className="w-full">
              <div className="mb-8 text-left">
                <h2 className="mb-4 font-bodoni text-3xl font-thin text-black md:text-4xl">
                  Everything You Need in RAG, and More
                </h2>
                <p className="font-sans text-lg text-gray-600">
                  MCP, RBAC, Multi-tenancy are directly baked in to Morphik.
                  Here are some features that we ❤️
                </p>
              </div>
              <BentoGridThirdDemo />
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      {/* <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center">
            <div className="w-full">
              <div className="rounded-lg bg-white p-16 text-center">
                <h2 className="font-bodoni text-3xl md:text-4xl font-thin text-black mb-6">
                  Ready to Build?
                </h2>
                <p className="font-sans text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
                  Join thousands of developers who trust Morphik for their most critical RAG applications.
                  Get started in minutes with our comprehensive documentation and support.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <button className="bg-black text-white font-sans px-8 py-3 rounded-md hover:bg-gray-800 transition-colors text-lg">
                    Start Building Now
                  </button>
                  <button className="border border-black text-black font-sans px-8 py-3 rounded-md hover:bg-gray-50 transition-colors text-lg">
                    View Documentation
                  </button>
                </div>
                <div className="mt-8 pt-8 border-t border-gray-200">
                  <p className="font-sans text-sm text-gray-500">
                    Free tier available • No credit card required • Deploy in 2 minutes
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section> */}

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white">
        <div className="container mx-auto px-4 py-20">
          {/* Main Footer Content */}
          <div className="mb-16 text-center">
            <h2 className="mb-6 font-bodoni text-4xl font-thin text-black md:text-5xl lg:text-6xl">
              Ready to Build?
            </h2>
            <p className="mx-auto mb-12 max-w-3xl font-sans text-xl text-gray-600">
              Join thousands of developers who trust Morphik for their most
              critical RAG applications. Get started in minutes with our
              comprehensive documentation and support.
            </p>

            <div className="mb-12 flex flex-col items-center justify-center gap-4 sm:flex-row">
              <a
                href="/signup"
                className="rounded-md bg-black px-8 py-3 font-sans text-lg text-white transition-colors hover:bg-gray-800"
              >
                Start Building Now
              </a>
              <a
                href="https://docs.morphik.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md border border-black px-8 py-3 font-sans text-lg text-black transition-colors hover:bg-gray-50"
              >
                View Documentation
              </a>
            </div>

            <div className="border-t border-gray-200 pt-8">
              <p className="font-sans text-base text-gray-500">
                Free tier available • No credit card required • Deploy in 2
                minutes
              </p>
            </div>
          </div>

          {/* Footer Links */}
          <div className="mb-12 grid grid-cols-1 gap-8 border-t border-gray-200 pt-12 sm:grid-cols-2 md:mb-16 md:grid-cols-4 md:gap-12 md:pt-16">
            <div>
              <h3 className="mb-4 font-sans text-base font-medium text-black md:mb-6 md:text-lg">
                Company
              </h3>
              <ul className="space-y-3 md:space-y-4">
                <li>
                  <a
                    href="/blog"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Blog
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:founders@morphik.ai"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Contact
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.ycombinator.com/companies/morphik"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    About
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/morphik-org/morphik-core"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Careers
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="mb-6 font-sans text-lg font-medium text-black">
                Products
              </h3>
              <ul className="space-y-4">
                <li>
                  <a
                    href="https://github.com/morphik-org/morphik-core"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Morphik Core
                  </a>
                </li>
                <li>
                  <a
                    href="/dashboard"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Morphik Cloud
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:founders@morphik.ai"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Enterprise
                  </a>
                </li>
                <li>
                  <a
                    href="/pricing"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Pricing
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="mb-6 font-sans text-lg font-medium text-black">
                Developers
              </h3>
              <ul className="space-y-4">
                <li>
                  <a
                    href="https://docs.morphik.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    API Documentation
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/morphik-org/morphik-core"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    GitHub
                  </a>
                </li>
                <li>
                  <a
                    href="https://docs.morphik.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Examples
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/morphik-org/morphik-core"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Community
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="mb-6 font-sans text-lg font-medium text-black">
                Resources
              </h3>
              <ul className="space-y-4">
                <li>
                  <a
                    href="/solutions"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Solutions
                  </a>
                </li>
                <li>
                  <a
                    href="https://docs.morphik.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Help Center
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:founders@morphik.ai"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Support
                  </a>
                </li>
                <li>
                  <a
                    href="/terms"
                    className="font-sans text-sm text-gray-600 transition-colors hover:text-black md:text-base"
                  >
                    Terms
                  </a>
                </li>
              </ul>
            </div>
          </div>

          {/* Footer Bottom */}
          <div className="flex flex-col items-center justify-between border-t border-gray-200 pt-6 md:flex-row md:pt-8">
            <div className="mb-4 flex items-center md:mb-0">
              <img
                src="/assets/morphikblack.png"
                alt="Morphik"
                className="mr-3 h-5 md:mr-4 md:h-6"
              />
              <p className="font-sans text-xs text-gray-500 md:text-sm">
                © 2025 Morphik — Built in San Francisco
              </p>
            </div>
            <div className="flex flex-col space-y-2 text-center sm:flex-row sm:space-x-6 sm:space-y-0">
              <a
                href="/privacy-policy.html"
                className="font-sans text-xs text-gray-500 transition-colors hover:text-black md:text-sm"
              >
                Privacy Policy
              </a>
              <a
                href="/terms"
                className="font-sans text-xs text-gray-500 transition-colors hover:text-black md:text-sm"
              >
                Terms of Service
              </a>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
```

## SpotlightComponent

```tsx
'use client';
import React, { useState, useEffect } from 'react';
import { highlight } from 'codehike/code';
import { tokenTransitions } from './code-hike/code';
import { Pre } from 'codehike/code';

const setupCode = `from morphik import Morphik
import os

db = Morphik(uri=os.getenv("MORPHIK_URI"))
`;

const ingestCode = `
doc = db.ingest_file("path/to/your/data")
doc.wait_for_completion()

print(doc.status)
# Output: completed
`;

const searchCode = `
response = db.query(
    "Why should I use Morphik?",
    llm_config={"model": "gpt-4-1"}, # configure any model here
    top_k=5,
)

print(response.completion)
# Output: It is both the fastest and most accurate way to store and
# search your data!
`;

const extractCode = `from morphik import Morphik
import os
from morphik.rules import MetadataExtractionRule
from pydantic import BaseModel

db = Morphik(uri=os.getenv("MORPHIK_URI"))

class DocumentInfo(BaseModel):
    title: str
    author: str
    department: str

doc = db.ingest_file(
    file="document.pdf",
    filename="document.pdf",
    metadata={"category": "research"},
    rules=[MetadataExtractionRule(schema=DocumentInfo)]
)
`;

const graphCode = `
graph = db.create_graph(
    name="RAG Information",
    filters={"topic": "RAG"},
)

response = db.query(
    "When should I use a knowledge graph for RAG?",
    graph_name="RAG Information"
)

print(response.completion)
# Output: Knowledge graphs are helpful for cross-document relationships.
# They are also helpful when you want to mix structured and unstructured
# data into a single knowledge base.
`;

const researchCode = `query = "Create a report on the the latest developments in AI"
response = db.agent_query(query) # this is a longer lived task
for obj in response.display_objects:
    match obj["type"], obj["content"]:
        case "text", text:
            print(text)
            print(f"Source: {obj["source"]}")
        case "image", url:
            convert_url_to_image(url).show()
            print(f"Source: {obj["source"]}")
        case _: # define your own displays here
            pass

# Alternatively, create a report
response.create_report("save/path/to/report.pdf")
`;

const steps = [
  {
    title: 'Ingest',
    subtitle: 'Add your data, effortlessly.',
    code: setupCode + ingestCode,
  },
  {
    title: 'Extract',
    subtitle: 'Pull out structured information.',
    code: extractCode,
  },
  {
    title: 'Query',
    subtitle: 'Find what you need, instantly.',
    code: setupCode + searchCode,
  },
  // {
  //   title: 'Workflows',
  //   subtitle: 'Automate your data processes.',
  //   code: `await morphik.workflows.run("your_workflow_id");`
  // },
  {
    title: 'Graphs',
    subtitle: 'Visualize your knowledge.',
    code: setupCode + graphCode,
  },
  {
    title: 'Research',
    subtitle: 'Unlock new insights.',
    code: setupCode + researchCode,
  },
];

export default function SpotlightComponent() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [highlightedCode, setHighlightedCode] = useState<Awaited<
    ReturnType<typeof highlight>
  > | null>(null);

  useEffect(() => {
    const highlightCode = async () => {
      const highlighted = await highlight(
        {
          value: steps[activeIndex].code,
          lang: 'python',
          meta: '',
        },
        'light-plus'
      );
      setHighlightedCode(highlighted);
    };
    highlightCode();
  }, [activeIndex]);

  return (
    <div className="flex flex-col rounded-lg border border-black md:flex-row">
      <div className="w-full border-b border-black p-3 md:w-1/3 md:border-b-0 md:border-r md:p-4">
        <div className="flex space-x-2 overflow-x-auto md:flex-col md:space-x-0 md:space-y-2">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`flex-shrink-0 cursor-pointer rounded-lg border border-black p-3 md:mb-2 md:p-4 ${
                activeIndex === index
                  ? ''
                  : 'border-b-4 border-l-4 border-black'
              }`}
              onClick={() => setActiveIndex(index)}
            >
              <h3 className="whitespace-nowrap font-bodoni text-base text-black md:text-xl">
                {step.title}
              </h3>
              <p className="hidden font-sans text-sm text-gray-600 md:block">
                {step.subtitle}
              </p>
            </div>
          ))}
        </div>
      </div>
      <div className="w-full overflow-x-auto p-2 md:w-2/3 md:p-4">
        {highlightedCode && (
          <div className="min-w-[600px] text-xs md:min-w-0 md:text-base">
            <Pre code={highlightedCode} handlers={[tokenTransitions]} />
          </div>
        )}
      </div>
    </div>
  );
}
```

## BenchmarkShowcase

```tsx
'use client';
import React from 'react';
import { ExternalLink } from 'lucide-react';
import Image from 'next/image';

interface BenchmarkData {
  name: string;
  score: string;
  scoreNumber: number;
  description: string;
  logo?: string;
}

const benchmarkData: BenchmarkData[] = [
  {
    name: 'Morphik',
    score: '95.6',
    scoreNumber: 95.56,
    description: '43 out of 45 questions correct',
    logo: '/assets/Morphik-Logo-Square.png',
  },
  {
    name: 'Custom Pipeline',
    score: '66.7',
    scoreNumber: 66.67,
    description: 'SOTA OCR + Layout Detection + LangChain',
    logo: '/assets/langchain-dark-8x.png',
  },
  {
    name: 'OpenAI GPT-4',
    score: '13.3',
    scoreNumber: 13.33,
    description: '6 out of 45 questions correct',
    logo: '/assets/OpenAI-black-monoblossom.png',
  },
];

export default function BenchmarkShowcase() {
  return (
    <div className="relative overflow-hidden rounded-lg border border-black bg-white">
      {/* Grid Background */}
      <div className="absolute inset-0 opacity-[0.02]">
        <div
          className="h-full w-full"
          style={{
            backgroundImage: `
            linear-gradient(to right, #000 1px, transparent 1px),
            linear-gradient(to bottom, #000 1px, transparent 1px)
          `,
            backgroundSize: '20px 20px',
          }}
        />
      </div>

      <div className="relative p-4 md:p-8">
        {/* Benchmark Bars */}
        <div className="mb-8 space-y-4">
          {benchmarkData.map((item, index) => (
            <div key={index} className="flex items-center">
              {/* Label Column */}
              <div className="flex w-24 items-center justify-end gap-1 pr-2 md:w-36 md:gap-2 md:pr-4">
                {item.logo ? (
                  <Image
                    src={item.logo}
                    alt={item.name}
                    width={24}
                    height={24}
                    className="w-auto md:h-8 md:w-8"
                  />
                ) : (
                  <div className="flex h-3 w-3 items-center justify-center bg-gray-300 text-xs font-bold text-white md:h-4 md:w-4">
                    {item.name.charAt(0)}
                  </div>
                )}
                <span className="whitespace-nowrap text-right font-mono text-xs uppercase tracking-wide text-black md:text-sm">
                  {item.name === 'Custom Pipeline'
                    ? 'Custom'
                    : item.name === 'OpenAI GPT-4'
                      ? 'OpenAI FS'
                      : item.name.toUpperCase()}
                </span>
              </div>

              {/* Bar Container */}
              <div className="relative h-6 flex-1 border-gray-300 md:h-8">
                {/* Performance Bar */}
                <div
                  className={`h-full ${index === 0 ? 'bg-black' : 'bg-gray-200'} flex items-center justify-end pr-2 md:pr-3`}
                  style={{ width: `${item.scoreNumber}%` }}
                >
                  <span
                    className={`font-mono text-xs font-semibold md:text-sm ${index === 0 ? 'text-white' : 'text-black'}`}
                  >
                    <span className="md:hidden">
                      {parseFloat(item.score) < 10
                        ? parseFloat(item.score).toFixed(1)
                        : Math.round(parseFloat(item.score))}
                    </span>
                    <span className="hidden md:inline">{item.score}</span>
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Scale Labels */}
        <div className="relative mb-6 md:mb-8">
          <div className="flex justify-between pl-24 font-mono text-xs text-gray-500 md:pl-36">
            <span>0</span>
            <span className="hidden sm:inline">20</span>
            <span className="hidden sm:inline">40</span>
            <span className="hidden sm:inline">60</span>
            <span className="hidden sm:inline">80</span>
            <span>100</span>
          </div>
          <div className="mt-2 text-center">
            <span className="font-sans text-xs text-gray-500">
              Benchmark performance (%)
            </span>
          </div>
        </div>

        {/* Performance Metrics Grid */}
        <div className="grid grid-cols-1 gap-4 border-t border-gray-200 pt-4 sm:grid-cols-3 sm:gap-8 sm:pt-6">
          <div>
            <div className="mb-1 font-sans text-xs uppercase tracking-wider text-gray-500">
              ACCURACY
            </div>
            <div className="font-mono text-xl text-black md:text-2xl">96%</div>
          </div>
          <div>
            <div className="mb-1 font-sans text-xs uppercase tracking-wider text-gray-500">
              RETRIEVAL LATENCY
            </div>
            <div className="font-mono text-xl text-black md:text-2xl">
              200ms
            </div>
          </div>
          <div>
            <div className="mb-1 font-sans text-xs uppercase tracking-wider text-gray-500">
              SCALE
            </div>
            <div className="font-mono text-xl text-black md:text-2xl">
              1M+
              <br className="sm:hidden" /> Documents
            </div>
          </div>
        </div>

        {/* Footer Note */}
        <div className="mb-6 mt-6 flex flex-col gap-4 border-t border-gray-200 pt-4 md:mb-8 md:mt-8 md:flex-row md:items-end md:justify-between md:pt-6">
          <div className="flex-1">
            <p className="font-sans text-xs text-gray-500">
              Evaluation performed on July 8, 2025 using questions from TLDC
              (The LLM Data Company).
            </p>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <a
              href="https://github.com/morphik-org/morphik-core/tree/main/evaluations/custom_eval"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center border border-black px-4 py-2 font-sans text-sm font-medium text-black transition-colors"
            >
              View on Github
              <ExternalLink className="ml-2 h-4 w-4" />
            </a>
            <a
              href="https://github.com/morphik-org/morphik-core/tree/main/evaluations/custom_eval"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center bg-black px-4 py-2 font-sans text-sm font-medium text-white transition-colors"
            >
              Tech Blog
              <ExternalLink className="ml-2 h-4 w-4" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
```

## WhyMorphikSection

```tsx
'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, Network, ExternalLink, Cloud } from 'lucide-react';

interface TabData {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  videoSrc?: string;
  link?: {
    text: string;
    url: string;
  };
  icon: React.ComponentType<Record<string, unknown>>;
}

const tabsData: TabData[] = [
  {
    id: 'visual',
    title: 'Visual-first Retrieval',
    subtitle: 'Diagram Intelligence',
    description:
      "Morphik directly embeds each page in your input into it's store, meaning no context is lost to imperfect parsing or processing techniques.",
    videoSrc: '/assets/mcp-demo-final-final.mp4#t=14',
    link: {
      text: 'See Morphik with 4o-mini beat gpt-o3',
      url: 'https://www.morphik.ai/docs/blogs/gpt-vs-morphik-multimodal',
    },
    icon: BrainCircuit,
  },
  {
    id: 'knowledge',
    title: 'Knowledge Graphs',
    subtitle: 'Connect concepts across your entire knowledge base',
    description:
      'Our knowledge graph technology automatically links related concepts, creating a rich network of information that makes search more intuitive and powerful.',
    videoSrc: '/assets/graphs-lp.mp4',
    link: {
      text: "Explore a graph we made from Paul Graham's essays",
      url: 'https://pggraph.streamlit.app',
    },
    icon: Network,
  },
  {
    id: 'onprem',
    title: 'On-prem',
    subtitle: 'Deploy anywhere, maintain full control',
    description:
      'Whether you need air-gapped security or prefer to run in the cloud, Morphik gives you the flexibility to deploy however you need.',
    link: {
      text: 'Check out our GitHub repository',
      url: 'https://github.com/morphik-org/morphik-core',
    },
    icon: Cloud,
  },
];

export default function WhyMorphikSection() {
  const [activeTab, setActiveTab] = useState('visual');
  const activeTabData =
    tabsData.find((tab) => tab.id === activeTab) || tabsData[0];

  return (
    <div className="w-full">
      <div className="mb-8">
        <div className="mb-6">
          <h2 className="mb-4 font-bodoni text-2xl font-thin text-black md:text-3xl lg:text-4xl">
            Why Morphik?
          </h2>
        </div>
        <div>
          <p className="font-sans text-base leading-relaxed text-gray-600 md:text-lg">
            Our search consistently outperforms other providers while being
            faster and cheaper to deploy. We excel at technical and{' '}
            <span
              className="cursor-pointer font-bold hover:underline"
              onClick={() => setActiveTab('knowledge')}
            >
              domain-specific search
            </span>
            . Morphik connects with any data source and ingests your knowledge
            in its native format - meaning perfect search over{' '}
            <span
              className="cursor-pointer font-bold hover:underline"
              onClick={() => setActiveTab('visual')}
            >
              complex diagrams
            </span>
            , schematics, and datasheets. We benefit heavily from{' '}
            <span
              className="cursor-pointer font-bold hover:underline"
              onClick={() => setActiveTab('onprem')}
            >
              open-source
            </span>
            : if you need a feature, open an issue or - better yet - submit a
            PR!
          </p>
        </div>
      </div>

      {/* Browser-like container */}
      <div className="rounded-lg border border-black">
        {/* Browser header with controls only */}
        {/* <div className="flex items-center p-3 border-b border-black bg-white"> */}
        {/* Browser controls */}
        {/* <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div> */}
        {/* </div> */}

        {/* Tab navigation */}
        <div className="flex overflow-x-auto border-b border-black">
          {tabsData.map((tab) => {
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative flex flex-shrink-0 items-center space-x-1 border-r border-black px-3 py-3 font-sans text-xs font-medium transition-all duration-200 md:space-x-2 md:px-4 md:text-sm ${
                  activeTab === tab.id
                    ? 'border-b-2 border-b-black text-black'
                    : 'text-black'
                } `}
              >
                <IconComponent className="h-3 w-3 md:h-4 md:w-4" />
                <span className="whitespace-nowrap">{tab.title}</span>
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-black"
                    initial={false}
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div className="p-4 md:p-8">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="flex flex-col items-start gap-6 md:flex-row md:gap-8"
          >
            {/* Text content */}
            <div className="space-y-4 md:w-1/3">
              <h3 className="font-bodoni text-xl font-thin text-black md:text-2xl">
                {activeTabData.subtitle}
              </h3>

              <p className="font-sans text-sm leading-relaxed text-black md:text-base">
                {activeTabData.description}
              </p>

              {activeTabData.link && (
                <a
                  href={activeTabData.link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 font-sans text-sm font-medium text-black hover:underline"
                >
                  <span>{activeTabData.link.text}</span>
                  <ExternalLink className="h-3 w-3 md:h-4 md:w-4" />
                </a>
              )}
            </div>

            {/* Video content or placeholder */}
            <div className="md:w-2/3">
              {activeTabData.videoSrc ? (
                <div className="overflow-hidden rounded-lg border border-black shadow-sm">
                  <video
                    key={`${activeTab}-video`}
                    className="h-auto w-full"
                    controls
                    autoPlay
                    muted
                    loop
                  >
                    <source src={activeTabData.videoSrc} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
              ) : (
                <div className="flex h-64 items-center justify-center overflow-hidden rounded-lg border border-black bg-white shadow-sm">
                  <div className="text-center text-black">
                    {React.createElement(activeTabData.icon, {
                      className: 'w-12 h-12 mx-auto mb-4 text-black',
                    })}
                    <p className="font-sans text-sm">Feature demonstration</p>
                    <p className="font-sans text-xs">Coming soon</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
```

## BentoGridThirdDemo

```tsx
'use client';
import { cn } from '@/lib/utils';
import React from 'react';
import Image from 'next/image';
import { BentoGrid, BentoGridItem } from './ui/bento-grid';
import {
  IconBoxAlignRightFilled,
  IconFileBroken,
  IconSignature,
  IconTableColumn,
} from '@tabler/icons-react';
import { highlight } from 'codehike/code';
import { tokenTransitions } from './code-hike/code';
import { Pre } from 'codehike/code';

export function BentoGridThirdDemo() {
  return (
    <BentoGrid className="md:auto-cols-1 mx-auto md:auto-rows-[20rem]">
      {items.map((item, i) => (
        <BentoGridItem
          key={i}
          title={item.title}
          description={item.description}
          header={item.header}
          className={cn('[&>p:text-lg]', item.className, 'border border-black')}
          icon={item.icon}
        />
      ))}
    </BentoGrid>
  );
}

const SkeletonOne = () => {
  const [highlightedCode, setHighlightedCode] = React.useState<Awaited<
    ReturnType<typeof highlight>
  > | null>(null);
  const jsonString = React.useMemo(
    () =>
      JSON.stringify(
        {
          mcpServers: {
            morphik: {
              command: 'npx',
              args: ['-y', '@morphik/mcp', '--uri=your-morphik-server-uri'],
            },
          },
        },
        null,
        2
      ),
    []
  );

  React.useEffect(() => {
    async function highlightCode() {
      const highlighted = await highlight(
        {
          value: jsonString,
          lang: 'json',
          meta: '',
        },
        'light-plus'
      );
      setHighlightedCode(highlighted);
    }
    highlightCode();
  }, [jsonString]);

  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(jsonString).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      },
      (err) => {
        console.error('Could not copy text: ', err);
      }
    );
  };

  return (
    <div className="relative flex h-full min-h-[6rem] w-full flex-1 overflow-hidden rounded-lg font-mono text-xs">
      <button
        onClick={copyToClipboard}
        className="absolute right-2 top-2 z-10 rounded-md bg-gray-100 px-2 py-1 text-gray-800 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 focus:ring-offset-gray-50"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
      {highlightedCode && (
        <Pre
          code={highlightedCode}
          handlers={[tokenTransitions]}
          className="h-full w-full"
        />
      )}
    </div>
  );
};
const items = [
  {
    title: 'MCP Support',
    description: (
      <span className="font-sans text-sm">
        Use Morphik with Claude, Open Web UI, or any other MCP client.
      </span>
    ),
    header: <SkeletonOne />,
    className: 'md:col-span-1',
    icon: (
      <Image
        src="/company-logos/mcp.png"
        alt="MCP"
        width={200}
        height={200}
        className="h-6 w-6"
      />
    ),
  },
  {
    title: 'Built-in Multi-tenancy',
    description: (
      <span className="font-sans text-sm">
        Control MxN mappings of users to instances. Spin up new instances with a
        single command.
      </span>
    ),
    header: (
      <Image
        src="/assets/multi-tenancy.png"
        alt="MCP"
        width={1200}
        height={1200}
        className="w-1200 h-1200"
      />
    ),
    className: 'md:col-span-1',
    icon: <IconFileBroken className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'RBAC and Judgement-driven AC',
    description: (
      <span className="font-sans text-sm">
        Use RBAC or define natural language &quot;judgement&quot; rules to
        control access to your data.
      </span>
    ),
    header: (
      <Image
        src="/assets/RBAC.png"
        alt="MCP"
        width={1200}
        height={1200}
        className="w-1200 h-1200"
      />
    ),
    className: 'md:col-span-1',
    icon: <IconSignature className="h-4 w-4 text-neutral-500" />,
  },
  {
    title: 'Use via Code, or access the - completely embeddable - Web UI',
    description: (
      <span className="font-sans text-sm">
        Use our API or access Morphik directly via the web UI. The
        Morphik&apos;s web UI can be white-labelled and embedded directly in
        your own application.
      </span>
    ),
    header: (
      <div className="h-48 w-full overflow-hidden rounded-lg">
        <video
          src="/vids/chat-ui.mp4"
          className="h-full w-full scale-150 object-cover"
          style={{ objectPosition: 'center top' }}
          controls
          autoPlay
          muted
          preload="metadata"
        />
      </div>
    ),
    className: 'md:col-span-2',
    icon: <IconTableColumn className="h-4 w-4 text-neutral-500" />,
  },

  {
    title: 'Run On-Prem, or in the Cloud',
    description: (
      <span className="font-sans text-sm">
        Morphik can be deployed on-prem or in the cloud. We support AWS, GCP,
        Azure, and more.
      </span>
    ),
    header: (
      <Image
        src="/assets/on-prem.png"
        alt="MCP"
        width={1200}
        height={1200}
        className="w-1200 h-1200 grayscale"
      />
    ),
    className: 'md:col-span-1',
    icon: <IconBoxAlignRightFilled className="h-4 w-4 text-neutral-500" />,
  },
];
```

## Design Principles

Based on the landing page code, here are the key design principles to follow:

### Typography
- **Headings**: Use `font-bodoni text-4xl font-thin` for main headings, with responsive sizes (`md:text-4xl lg:text-5xl`)
- **Body Text**: Use `font-sans text-base text-gray-600` for body text, with larger sizes on desktop (`md:text-lg lg:text-xl`)
- **Captions**: Use `font-sans text-sm text-gray-500` for smaller text
- **Monospace**: Use `font-mono` for code and technical content

### Colors
- **Primary**: Black (`text-black`, `bg-black`, `border-black`)
- **Text**: Gray-600 for body text (`text-gray-600`)
- **Subtle**: Gray-500 for captions (`text-gray-500`)
- **Background**: White (`bg-white`)

### Buttons
- **Primary**: `rounded-md bg-black px-6 py-3 text-center font-sans text-white transition-colors hover:bg-gray-800`
- **Secondary**: `rounded-md border border-black px-6 py-3 text-center font-sans text-black transition-colors hover:bg-gray-50`

### Layout
- **Container**: `container mx-auto px-4`
- **Sections**: `py-16` for vertical spacing
- **Borders**: Use `border border-black` consistently
- **Rounded corners**: `rounded-lg` for most containers

### Interactive Elements
- **Hover states**: Always include `transition-colors` and appropriate hover styles
- **Focus states**: Include proper focus states for accessibility
- **Active states**: Use clear visual indicators for active tabs/buttons

### Responsive Design
- Mobile-first approach with `md:` and `lg:` breakpoints
- Flexible layouts that stack on mobile and expand on desktop
- Appropriate text sizing for different screen sizes
