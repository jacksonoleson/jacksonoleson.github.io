(() => {
  const canvas = document.querySelector('.neural-network');
  if (!canvas) {
    const button = document.querySelector('.template-theme-toggle');
    if (!button) return;
    const key = 'ai-portfolio-color-mode';
    const scheme = window.matchMedia('(prefers-color-scheme: dark)');
    try {
      const mode = window.localStorage.getItem(key);
      if (mode === 'light' || mode === 'dark') document.documentElement.dataset.colorMode = mode;
    } catch (_) { /* System color preference remains available. */ }
    const updateButton = () => {
      const dark = document.documentElement.dataset.colorMode === 'dark'
        || (!document.documentElement.dataset.colorMode && scheme.matches);
      button.dataset.mode = dark ? 'dark' : 'light';
      button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    };
    button.addEventListener('click', () => {
      const dark = document.documentElement.dataset.colorMode === 'dark'
        || (!document.documentElement.dataset.colorMode && scheme.matches);
      const mode = dark ? 'light' : 'dark';
      document.documentElement.dataset.colorMode = mode;
      try { window.localStorage.setItem(key, mode); } catch (_) { /* This is a visual preference only. */ }
      updateButton();
    });
    scheme.addEventListener('change', updateButton);
    updateButton();
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) return;

  const container = canvas.parentElement;
  const isPageBackground = canvas.classList.contains('page-network');
  const isAiPortfolio = document.body.classList.contains('theme-ai-portfolio');
  const controls = document.querySelector('.animation-controls');
  const select = document.querySelector('#animation-style');
  const pauseButton = document.querySelector('.animation-pause');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colorScheme = window.matchMedia('(prefers-color-scheme: dark)');
  const storageKey = 'profile-animation';
  const colorModeKey = 'ai-portfolio-color-mode';
  const colorModeButton = document.querySelector('.template-theme-toggle');
  let width = 0;
  let height = 0;
  let frame;
  let lastTime;
  let inView = true;
  let accent;
  let mono;
  let paused = false;
  let style = 'neural';
  let updateColorModeButton = () => {};

  // Each style owns its particles; the shared controller owns motion and sizing.
  // Add a factory here and an option in the selector to collect more styles.
  const styles = new Map([
    ['neural', createNeuralNetwork],
    ['starfield', createStarfield],
    ['binary', createBinaryFlow],
    ['off', () => ({ resize() {}, update() {}, draw() {} })]
  ]);

  try {
    const saved = JSON.parse(window.localStorage.getItem(storageKey));
    if (saved && styles.has(saved.style)) {
      style = saved.style;
      paused = saved.paused === true;
    }
  } catch (_) { /* Preferences are optional when storage is unavailable. */ }

  // The AI portfolio deliberately uses its neural field; experimental modes
  // remain available on the other templates without changing this visual shell.
  if (document.body.classList.contains('theme-ai-portfolio')) {
    try {
      const mode = window.localStorage.getItem(colorModeKey);
      if (mode === 'light' || mode === 'dark') document.documentElement.dataset.colorMode = mode;
    } catch (_) { /* System color preference remains available. */ }
    style = document.documentElement.dataset.colorMode === 'dark'
      || (!document.documentElement.dataset.colorMode && colorScheme.matches)
      ? 'starfield'
      : 'neural';
    paused = false;
  }

  let animation = styles.get(style)();

  function createNeuralNetwork() {
    const nodes = [];
    const clusters = [];
    return {
      resize() {
        nodes.length = 0;
        clusters.length = 0;
        const clusterCount = isAiPortfolio ? 1 : 3;
        for (let index = 0; index < clusterCount; index += 1) {
          clusters.push({
            x: isAiPortfolio ? width * .85 : width * (.18 + Math.random() * .64),
            y: isAiPortfolio ? height * .31 : height * (.18 + Math.random() * .64),
            vx: (Math.random() - .5) * .05,
            vy: (Math.random() - .5) * .05,
            fixed: isAiPortfolio
          });
        }
        const count = Math.max(24, Math.min(48, Math.round((width * height) / 4800)));
        for (let index = 0; index < count; index += 1) {
          nodes.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - .5) * .18,
            vy: (Math.random() - .5) * .18,
            radius: 1.2 + Math.random() * 1.5
          });
        }
      },
      update(delta) {
        clusters.forEach((cluster) => {
          if (cluster.fixed) {
            const profileText = document.querySelector('.profile-text')?.getBoundingClientRect();
            const profile = document.querySelector('.profile')?.getBoundingClientRect();
            if (profileText && profile) {
              cluster.x = Math.min(width - 110, profileText.right + (width - profileText.right) * .28);
              cluster.y = profileText.top - profile.top + profileText.height / 2 + 38;
            }
            return;
          }
          cluster.x += cluster.vx * delta * 60;
          cluster.y += cluster.vy * delta * 60;
          if (cluster.x < width * .1 || cluster.x > width * .9) cluster.vx *= -1;
          if (cluster.y < height * .1 || cluster.y > height * .9) cluster.vy *= -1;
        });
        nodes.forEach((node) => {
          const nearestCluster = clusters.reduce((nearest, cluster) => {
            const nearestDistance = Math.hypot(node.x - nearest.x, node.y - nearest.y);
            const clusterDistance = Math.hypot(node.x - cluster.x, node.y - cluster.y);
            return clusterDistance < nearestDistance ? cluster : nearest;
          });
          node.vx += (nearestCluster.x - node.x) * .000025 * delta * 60;
          node.vy += (nearestCluster.y - node.y) * .000025 * delta * 60;
          const speed = Math.hypot(node.vx, node.vy);
          if (speed > .6) {
            node.vx = node.vx / speed * .6;
            node.vy = node.vy / speed * .6;
          }
          // Preserve the original speed at 60 Hz across other refresh rates.
          node.x += node.vx * delta * 60;
          node.y += node.vy * delta * 60;
          if (node.x < 0 || node.x > width) node.vx *= -1;
          if (node.y < 0 || node.y > height) node.vy *= -1;
        });
      },
      draw() {
        const reach = Math.min(145, Math.max(95, width * .22));
        context.strokeStyle = accent;
        context.lineWidth = .8;
        for (let first = 0; first < nodes.length; first += 1) {
          for (let second = first + 1; second < nodes.length; second += 1) {
            const distance = Math.hypot(nodes[first].x - nodes[second].x, nodes[first].y - nodes[second].y);
            if (distance > reach) continue;
            context.beginPath();
            context.moveTo(nodes[first].x, nodes[first].y);
            context.lineTo(nodes[second].x, nodes[second].y);
            const proximity = 1 - distance / reach;
            context.globalAlpha = .05 + .3 * proximity * proximity;
            context.stroke();
          }
        }
        context.globalAlpha = .78;
        context.fillStyle = accent;
        nodes.forEach((node) => {
          context.beginPath();
          context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          context.fill();
        });
      }
    };
  }

  function createBinaryFlow() {
    const digits = [];
    const far = 3.6;
    const near = .38;

    function seed(digit, initial = false) {
      digit.x = (Math.random() - .5) * 2.7;
      digit.y = (Math.random() - .5) * 2.7;
      digit.z = initial ? near + Math.random() * (far - near) : far;
      digit.value = Math.random() < .5 ? '0' : '1';
      digit.speed = .12 + Math.random() * .12;
      digit.phase = Math.random() * Math.PI * 2;
    }

    return {
      resize() {
        digits.length = 0;
        const count = Math.max(40, Math.min(90, Math.round(width * height / 2800)));
        for (let index = 0; index < count; index += 1) {
          const digit = {};
          seed(digit, true);
          digits.push(digit);
        }
      },
      update(delta) {
        digits.forEach((digit) => {
          digit.z -= digit.speed * delta;
          digit.phase += delta * .18;
          if (digit.z < near) seed(digit);
        });
      },
      draw() {
        const glow = context.createRadialGradient(width * .6, height * .48, 0, width * .6, height * .48, Math.max(width, height) * .65);
        glow.addColorStop(0, accent);
        glow.addColorStop(1, 'transparent');
        context.globalAlpha = .07;
        context.fillStyle = glow;
        context.fillRect(0, 0, width, height);
        context.fillStyle = accent;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.shadowColor = accent;

        // Project a field of digits toward the viewer, fading at both ends
        // so recycling a particle never creates a visible jump.
        digits.sort((a, b) => b.z - a.z).forEach((digit) => {
          const scale = 1 / digit.z;
          const x = width * .56 + digit.x * width * .62 * scale;
          const y = height * .5 + (digit.y + Math.sin(digit.phase) * .035) * height * .8 * scale;
          const size = Math.min(76, 23 * scale);
          if (x < -size || x > width + size || y < -size || y > height + size) return;
          const fadeIn = Math.min(1, (far - digit.z) / .65);
          const fadeOut = Math.min(1, (digit.z - near) / .4);
          context.globalAlpha = fadeIn * fadeOut * (.13 + Math.min(1, scale) * .2);
          context.font = `400 ${size}px ${mono}`;
          context.shadowBlur = Math.min(14, size * .22);
          context.fillText(digit.value, x, y);
        });
      }
    };
  }

  function createStarfield() {
    const stars = [];
    return {
      resize() {
        stars.length = 0;
        const count = Math.max(70, Math.min(150, Math.round(width * height / 4200)));
        for (let index = 0; index < count; index += 1) {
          stars.push({
            x: Math.random() * width,
            y: Math.random() * height,
            size: .45 + Math.random() * 1.45,
            speed: .06 + Math.random() * .24,
            phase: Math.random() * Math.PI * 2
          });
        }
      },
      update(delta) {
        stars.forEach((star) => {
          star.y += star.speed * delta * 60;
          star.phase += delta * (1.1 + star.speed * 2);
          if (star.y > height + 3) {
            star.y = -3;
            star.x = Math.random() * width;
          }
        });
      },
      draw() {
        context.fillStyle = '#81f5a7';
        context.strokeStyle = '#52d982';
        context.shadowColor = '#5dff91';
        context.shadowBlur = 7;
        stars.forEach((star) => {
          const brightness = .18 + (Math.sin(star.phase) + 1) * .16;
          context.globalAlpha = brightness * .5;
          context.lineWidth = Math.max(.45, star.size * .55);
          context.beginPath();
          context.moveTo(star.x, star.y - star.speed * 34);
          context.lineTo(star.x, star.y);
          context.stroke();
          context.globalAlpha = brightness;
          context.beginPath();
          context.arc(star.x, star.y, star.size, 0, Math.PI * 2);
          context.fill();
        });
        context.shadowBlur = 0;
      }
    };
  }

  function readColors() {
    const computed = getComputedStyle(document.body);
    accent = computed.getPropertyValue('--accent').trim() || '#3159b8';
    mono = computed.getPropertyValue('--mono').trim() || 'monospace';
  }

  function draw() {
    context.clearRect(0, 0, width, height);
    context.save();
    if (isAiPortfolio) {
      context.beginPath();
      context.rect(0, 0, width, height);
      context.clip();
    }
    animation.draw();
    context.restore();
    if (isPageBackground && !isAiPortfolio) {
      const bubbleSelector = document.body.classList.contains('theme-ai-portfolio')
        ? '.publication-card, .post-list li, .post, .experience-card'
        : '.profile, .publication-card, .post-list li, .post, .experience-card';
      document.querySelectorAll(bubbleSelector).forEach((bubble) => {
        const bounds = bubble.getBoundingClientRect();
        context.clearRect(bounds.left, bounds.top, bounds.width, bounds.height);
      });
    }
  }

  function resize() {
    const hero = isAiPortfolio ? document.querySelector('.profile') : null;
    const bounds = hero
      ? { width: window.innerWidth, height: hero.offsetHeight }
      : isPageBackground
      ? { width: window.innerWidth, height: window.innerHeight }
      : container.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = bounds.width;
    height = bounds.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (hero) canvas.style.top = `${hero.offsetTop}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    animation.resize();
    draw();
  }

  function animate(time) {
    const delta = lastTime === undefined ? 0 : Math.min((time - lastTime) / 1000, .05);
    lastTime = time;
    animation.update(delta);
    draw();
    frame = window.requestAnimationFrame(animate);
  }

  function updateMotion() {
    window.cancelAnimationFrame(frame);
    lastTime = undefined;
    draw();
    if (pauseButton) {
      pauseButton.disabled = style === 'off' || reduceMotion.matches;
      pauseButton.textContent = reduceMotion.matches ? 'Static' : paused ? 'Play' : 'Pause';
      pauseButton.setAttribute('aria-pressed', String(paused));
      pauseButton.setAttribute('aria-label', reduceMotion.matches ? 'Animation disabled by reduced motion preference' : paused ? 'Play background animation' : 'Pause background animation');
    }
    if (!paused && !reduceMotion.matches && !document.hidden && inView && style !== 'off') {
      frame = window.requestAnimationFrame(animate);
    }
  }

  function savePreference() {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify({ style, paused }));
    } catch (_) { /* Switching still works without persistent storage. */ }
  }

  if (controls && select && pauseButton) {
    select.value = style;
    select.addEventListener('change', () => {
      if (!styles.has(select.value)) return;
      style = select.value;
      animation = styles.get(style)();
      animation.resize();
      savePreference();
      updateMotion();
    });
    pauseButton.addEventListener('click', () => {
      paused = !paused;
      savePreference();
      updateMotion();
    });
    controls.hidden = false;
  }

  if (colorModeButton) {
    let mode;
    try { mode = window.localStorage.getItem(colorModeKey); } catch (_) { /* System mode remains available. */ }
    if (mode === 'light' || mode === 'dark') document.documentElement.dataset.colorMode = mode;
    updateColorModeButton = () => {
      const dark = document.documentElement.dataset.colorMode === 'dark'
        || (!document.documentElement.dataset.colorMode && colorScheme.matches);
      colorModeButton.dataset.mode = dark ? 'dark' : 'light';
      colorModeButton.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
      const nextStyle = dark ? 'starfield' : 'neural';
      if (isAiPortfolio && style !== nextStyle) {
        style = nextStyle;
        animation = styles.get(style)();
        if (width && height) animation.resize();
      }
      readColors();
      draw();
    };
    colorModeButton.addEventListener('click', () => {
      const currentlyDark = document.documentElement.dataset.colorMode === 'dark'
        || (!document.documentElement.dataset.colorMode && colorScheme.matches);
      const nextMode = currentlyDark ? 'light' : 'dark';
      document.documentElement.dataset.colorMode = nextMode;
      try { window.localStorage.setItem(colorModeKey, nextMode); } catch (_) { /* This is a visual preference only. */ }
      updateColorModeButton();
    });
    updateColorModeButton();
  }

  readColors();
  if (isPageBackground) {
    window.addEventListener('resize', resize, { passive: true });
  } else {
    new ResizeObserver(resize).observe(container);
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      updateMotion();
    }).observe(container);
  }
  reduceMotion.addEventListener('change', updateMotion);
  colorScheme.addEventListener('change', () => {
    updateColorModeButton();
  });
  document.addEventListener('visibilitychange', updateMotion);
  resize();
  updateMotion();
})();
