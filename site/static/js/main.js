(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };

  /* ---------- Method loop ---------- */
  var steps = [
    ['01 · Ground', 'Start from a real scene.',
     'An open-vocabulary task is matched against an offline repository of real-robot recordings in a shared embedding space, and a semantically aligned initial observation is sampled. Simulation starts from a scene the robot has actually seen.',
     'Output: a grounded state o₀ for the task ℓ.'],
    ['02 · Route', 'Choose the right expert.',
     'A high-level router, conditioned on the scene, the instruction and a long-term experience memory, selects a VLA expert from the skill library: GR00T-N1.6, π₀.₅, π₀ or π₀-FAST.',
     'Output: the selected expert πₘ.'],
    ['03 · Imagine', 'Test it in the world model.',
     'The expert acts inside an action-conditioned video world model, which predicts the rollout o₀ … o_H. Counterfactual rollouts score candidate experts before anything moves in the real world.',
     'Output: an imagined trajectory τ.'],
    ['04 · Evolve', 'Learn from the diagnosis.',
     'A vision-language evaluator scores the rollout and tags the cause of failure: world-model failure, policy failure or task-scenery mismatch. Only the part that failed is updated, instead of retraining everything.',
     'Updates: world model, router memory or a specific expert.']
  ];
  var cur = 0;
  var nodes = document.querySelectorAll('.loop .node');
  function setStep(i) {
    cur = (i + 4) % 4;
    var s = steps[cur];
    $('step-num').textContent = s[0]; $('step-title').textContent = s[1];
    $('step-text').textContent = s[2]; $('step-upd').textContent = s[3];
    nodes.forEach(function (n, k) { n.classList.toggle('on', k === cur); n.setAttribute('aria-pressed', k === cur); });
    var c = 2 * Math.PI * 160;
    $('loop-arc').setAttribute('stroke-dasharray', (c * (cur + 1) / 4) + ' ' + c);
  }
  nodes.forEach(function (n) {
    n.addEventListener('click', function () { setStep(+n.dataset.step); });
    n.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setStep(+n.dataset.step); } });
  });
  $('step-prev').addEventListener('click', function () { setStep(cur - 1); });
  $('step-next').addEventListener('click', function () { setStep(cur + 1); });
  setStep(0);

  /* ---------- Demo tabs ---------- */
  var tabs = document.querySelectorAll('[role="tab"]');
  tabs.forEach(function (t, i) {
    t.tabIndex = t.getAttribute('aria-selected') === 'true' ? 0 : -1;
    t.addEventListener('keydown', function (e) {
      var next = e.key === 'ArrowRight' ? (i + 1) % tabs.length : e.key === 'ArrowLeft' ? (i + tabs.length - 1) % tabs.length : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : -1;
      if (next >= 0) { e.preventDefault(); tabs[next].focus(); tabs[next].click(); }
    });
    t.addEventListener('click', function () {
      tabs.forEach(function (x) {
        var on = x === t; x.setAttribute('aria-selected', on); x.tabIndex = on ? 0 : -1;
        $(x.getAttribute('aria-controls')).hidden = !on;
      });
      document.querySelectorAll('video').forEach(function (v) { if (v.id !== 'film-video') v.pause(); });
      observeAuto();
    });
  });

  /* ---------- Real-robot tasks (values from the per-task success figure) ---------- */
  var tasks = [
    ['Open lid', 5, 45, 'demo_real_open_lid.jpg'],
    ['Reorient', 40, 95, null],
    ['Pick on', 55, 90, 'demo_real_pick_on.jpg'],
    ['Wipe board', 15, 65, 'demo_real_wipe_board.jpg'],
    ['Fold towel', 20, 55, 'demo_real_fold_towel.jpg']
  ];
  $('tasks').innerHTML = tasks.map(function (t) {
    return '<li class="task"><div class="task__top"><span class="task__name">' + t[0] + '</span><span class="task__nums">' + t[1] + '% → <b>' + t[2] + '%</b></span></div>' +
      '<div class="track" aria-hidden="true"><i style="width:' + t[1] + '%"></i><b style="width:' + t[2] + '%"></b></div>' +
      (t[3] ? '<img src="static/images/' + t[3] + '" alt="Frames from the ' + t[0].toLowerCase() + ' task" loading="lazy">' : '') + '</li>';
  }).join('');

  /* ---------- Results ---------- */
  var skills = ['S1 Pick-and-place', 'S2 Tool-use', 'S3 Reorientation', 'S4 Deformable', 'S5 Articulation'];
  var data = {
    'GR00T-N1.6': [[.38, .24, .18, .28, .20], [.48, .50, .46, .42, .48], 25.6, 46.8, '+82.8%'],
    'π₀.₅': [[.32, .14, .24, .18, .28], [.48, .28, .38, .44, .48], 23.2, 41.2, '+77.6%'],
    'π₀': [[.40, .22, .26, .32, .22], [.62, .38, .40, .40, .50], 28.4, 46.0, '+62.0%'],
    'π₀-FAST': [[.38, .18, .26, .24, .26], [.66, .38, .44, .36, .42], 26.4, 45.2, '+71.2%']
  };
  var MAX = 70; // axis max, percent
  function bar(cls, v) { return '<div class="cbar ' + cls + '" style="width:' + (v / MAX * 88) + '%"><span>' + v.toFixed(1) + '%</span></div>'; }
  var names = Object.keys(data);
  $('bb-pills').innerHTML = names.map(function (n, i) { return '<button type="button" aria-pressed="' + (i === 0) + '">' + n + '</button>'; }).join('');
  $('skill-chart').innerHTML = skills.map(function (s, i) {
    var p = s.split(' '); return '<div class="crow"><span class="lbl">' + p[0] + '<small>' + p.slice(1).join(' ') + '</small></span><div class="cbars"><div class="cbar base" id="sb' + i + '"><span></span></div><div class="cbar evo" id="se' + i + '"><span></span></div></div></div>';
  }).join('');
  function showBackbone(n) {
    var d = data[n];
    skills.forEach(function (s, i) {
      var b = $('sb' + i), e = $('se' + i);
      b.style.width = (d[0][i] * 100 / MAX * 88) + '%'; b.firstChild.textContent = Math.round(d[0][i] * 100) + '%';
      e.style.width = (d[1][i] * 100 / MAX * 88) + '%'; e.firstChild.textContent = Math.round(d[1][i] * 100) + '%';
    });
    $('bb-avg').innerHTML = 'Average: ' + d[2].toFixed(1) + '% → <b>' + d[3].toFixed(1) + '%</b>';
    $('bb-gain').innerHTML = 'Relative gain <b>' + d[4] + '</b>';
  }
  document.querySelectorAll('#bb-pills button').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('#bb-pills button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); });
      showBackbone(b.textContent);
    });
  });
  showBackbone(names[0]);
  $('avg-chart').innerHTML = names.map(function (n) {
    var d = data[n]; return '<div class="crow"><span class="lbl">' + n + '</span><div class="cbars">' + bar('base', d[2]) + bar('evo', d[3]) + '</div></div>';
  }).join('');

  /* ---------- Muted rollouts play while visible ---------- */
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var io = null;
  function observeAuto() {
    var vids = document.querySelectorAll('video[data-auto]');
    if (reduce) { vids.forEach(function (v) { v.setAttribute('controls', ''); }); return; }
    if (!('IntersectionObserver' in window)) { vids.forEach(function (v) { v.setAttribute('controls', ''); }); return; }
    if (!io) io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting && !v.closest('[hidden]')) { v.preload = 'auto'; var p = v.play(); if (p && p.catch) p.catch(function () { v.setAttribute('controls', ''); }); }
        else v.pause();
      });
    }, { threshold: .35 });
    vids.forEach(function (v) { io.unobserve(v); io.observe(v); });
  }
  observeAuto();

  /* ---------- One video with sound controls at a time ---------- */
  document.querySelectorAll('video[controls]:not([data-auto])').forEach(function (v) {
    v.addEventListener('play', function () {
      document.querySelectorAll('video[controls]:not([data-auto])').forEach(function (o) { if (o !== v) o.pause(); });
    });
  });

  /* ---------- Copy BibTeX ---------- */
  $('copy').addEventListener('click', function () {
    var btn = this, text = $('bib').textContent;
    function ok() { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); }
    function fallback() { var r = document.createRange(); r.selectNodeContents($('bib')); var s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = 'Press ⌘C / Ctrl+C'; }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(ok, fallback); else fallback();
  });
})();
