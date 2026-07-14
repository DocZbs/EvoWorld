const copyButton = document.querySelector('#copy-citation');
const copyStatus = document.querySelector('#copy-status');
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    copyStatus.textContent = 'Citation copied.';
  } catch {
    copyStatus.textContent = 'Copy is unavailable. Select the citation text below and copy it manually.';
  }
});
// Visible without JavaScript; one-shot motion is a progressive enhancement.
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
if (!reducedMotion.matches && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('entered');
        observer.unobserve(entry.target);
      }
    }
  }, { threshold: 0.1 });
  document.querySelectorAll('.method-steps li, .result-panel, .demo-grid figure').forEach((item) => observer.observe(item));
  reducedMotion.addEventListener('change', (event) => { if (event.matches) observer.disconnect(); });
}
// Do not compete for audio/attention when a visitor chooses another clip.
document.querySelectorAll('video').forEach((video) => {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach((other) => { if (other !== video) other.pause(); });
  });
});
