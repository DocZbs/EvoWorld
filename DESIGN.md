---
version: alpha
name: EvoWorld research project
description: A compact blue-and-white research editorial with classical typography and real demonstration footage.
colors:
  paper: '#f6f8f7'
  ink: '#152f4b'
  blue: '#235bb0'
  gold: '#d9ac55'
  muted: '#53687d'
  line: '#c8d3dd'
  wash: '#e8eef3'
  white: '#ffffff'
  base: '#9caebe'
  scroll-thumb: '#7a91a6'
  scroll-hover: '#53687d'
  scroll-active: '#152f4b'
typography:
  display:
    fontFamily: "Georgia, 'Times New Roman', serif"
  sans:
    fontFamily: "'Avenir Next', 'Segoe UI', Arial, sans-serif"
  mono:
    fontFamily: "'Courier New', monospace"
spacing:
  section: 64px
  page-max: 1240px
rounded:
  DEFAULT: 0px
components:
  video: {}
  button: {}
  disclosure: {}
  citation: {}
---
# EvoWorld design context

## Audience and register
A public research content site for robotics researchers and collaborators. English scientific content, international audience. No Japanese locale or Japan-market flows. The user's personal homepage establishes blue/white classical Greek restraint, compact information density and subtle motion. This redesign carries those choices into a research-focused presentation.

## Visual signature
Monumental EvoWorld serif wordmark paired with the real research film. Quiet gold rules and white/blue planes recall classical proportion without ornamental illustrations competing with experiments. No gradients, unrelated stock photos or generated robot demonstrations.

## Runtime ownership
`site/static/css/style.css :root` owns runtime colors, fonts and spacing. The frontmatter mirrors those tokens; there are no adapters. Section spacing is 64px desktop and 42px mobile. Shared video, button, disclosure and table styles apply throughout. Native video controls deliberately retain platform playback/fullscreen behavior.

## Content hierarchy
Short thesis, film, complete title, expandable author affiliations, three headline findings, four-step method, real and imagined demos, results, expandable full tables and citation. Retain all scientific data and original authors. Table 1 supports mean 25.9%→44.8%; the local rebuttal supports real-robot 29%→70% with five tasks and 20 trials each. Keep these settings visibly distinct.

## Interaction and accessibility
No autoplay. Videos use native keyboard-operable controls and only one plays at a time. Film includes descriptions track and visible transcript. Its original source room audio is omitted; explanation is on-screen. Label all speed changes and generated versus physical footage. Native details keep long content available without JavaScript. Clipboard failure provides inline recovery. Global scrollbar tokens, visible focus, reduced motion, stable media ratios, local assets, one-shot animation only. No analytics or third-party embeds.

## Video provenance
The film uses user-local EvoWorld recordings, not synthesized robot footage. Generated rollout sources retain all three camera views. Real towel recording uses the full first 115 seconds at 5×. Individual rollout players use original speed. Film edit is reproducible from `projects/evoworld/development/render_film.py` and `cut-list.json` in the original CV workspace. Video results cite the paper and rebuttal separately. All source recordings remain unchanged.

## Verification
Run the existing site release tests, syntax check, premium audit, desktop/mobile screenshots, playback and seeking, keyboard disclosures, clipboard success/failure, reduced motion, and deployment. The site-only Pages workflow remains the deployment owner.

## User-supplied October 2026 revision (current)
The owner's `EvoWorld-update/site` replaces the earlier visual direction: dark film-first hero, Lora display type, Source Sans 3 body, IBM Plex Mono utility, gold accents, warm paper and system dark mode. `site/static/css/style.css :root` and its theme overrides are now the canonical tokens, superseding the earlier frontmatter palette and spacing. The new owner-supplied film is 66 seconds with a matching poster and VTT captions. Existing PDF, diagrams and individual rollouts remain available. The method loop, demo tabs and backbone result selector are interactive. Native rollout controls allow pausing motion, and tabs support arrow/Home/End keyboard navigation.
