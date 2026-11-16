// Agentic Multimodal 3D Co-Pilot: Academic Tutor & Studio Technical Director.
// Course: 3D Computer Vision Course Project · IIIT Dharwad

import { api, clear, h, mount, state, toast } from './lib.js';

const ACADEMIC_PROMPTS = [
  'Explain epipolar geometry and depth loss',
  'What is Marching Cubes and the Euler characteristic?',
  'Compare NeRF vs 3D Gaussian Splatting (3DGS)',
  'Inspect depth tensor statistics',
];

const CREATIVE_PROMPTS = [
  'Reduce polycount to 5000 faces',
  'Repair mesh and make it watertight',
  'Convert model to STL for 3D printing',
  'How do I light and render this asset in Blender?',
];

export async function renderChat(params, container, { embedded = false, activeMeshId = null, onToolExecuted = null } = {}) {
  const messages = [];
  let active = true;
  let pending = false;
  let persona = 'academic'; // 'academic' | 'creative'
  state.routeCleanup = () => { active = false; };

  const personaSeg = h('div', { class: 'seg mb8', role: 'group', 'aria-label': 'Agent Persona' });
  const btnAcademic = h('button', { class: 'active', type: 'button' }, '🎓 Professor 3D');
  const btnCreative = h('button', { type: 'button' }, '🎨 Studio Director');
  mount(personaSeg, btnAcademic, btnCreative);

  const status = h('p', { class: 'muted mb8', style: 'font-size:12.5px;', role: 'status' }, 'Co-Pilot ready · Tools active');
  const transcript = h('div', { class: 'chat-transcript', role: 'log', 'aria-label': 'Conversation' });
  const input = h('textarea', {
    id: 'chat-message', rows: embedded ? '2' : '3', maxlength: '4000', required: true,
    placeholder: 'Ask theoretical CV questions or command tools (e.g. "reduce polycount to 5000")',
  });
  const send = h('button', { class: 'btn primary sm', type: 'submit' }, 'Send');
  const reset = h('button', { class: 'btn sm ghost', type: 'button', onclick: () => {
    messages.length = 0;
    clear(transcript);
    renderWelcome();
  } }, 'Clear');

  const chipsContainer = h('div', { class: 'chat-chips row wrap gap4 mb8' });

  function renderChips() {
    clear(chipsContainer);
    const list = persona === 'academic' ? ACADEMIC_PROMPTS : CREATIVE_PROMPTS;
    list.forEach(q => {
      const chip = h('button', {
        class: 'btn sm ghost', style: 'font-size:11.5px;padding:3px 8px;', type: 'button',
        onclick: () => { input.value = q; input.focus(); },
      }, q);
      chipsContainer.appendChild(chip);
    });
  }

  function setPersona(next) {
    persona = next;
    btnAcademic.classList.toggle('active', persona === 'academic');
    btnCreative.classList.toggle('active', persona === 'creative');
    status.textContent = persona === 'academic'
      ? 'Professor 3D active (Computer Vision Academic Tutor)'
      : 'Studio 3D Director active (Technical Art & Tool Execution)';
    renderChips();
  }

  btnAcademic.onclick = () => setPersona('academic');
  btnCreative.onclick = () => setPersona('creative');

  function renderWelcome() {
    const welcome = h('article', { class: 'chat-message chat-agent' },
      h('div', { class: 'spread mb4' },
        h('strong', { style: 'color:var(--cyan);' }, persona === 'academic' ? '🎓 Professor 3D' : '🎨 Studio Director'),
        h('small', { class: 'muted' }, 'Agentic AI Co-Pilot')),
      h('p', { class: 'mb4' },
        persona === 'academic'
          ? 'Welcome to the 3D Computer Vision Laboratory. Ask me about single-view geometry, loss formulations, marching cubes, or Gaussian Splatting.'
          : 'Welcome to the 3D Studio. I can autonomously repair meshes, decimate polygon counts, convert formats to STL/OBJ, and diagnose your 3D assets.'));
    transcript.appendChild(welcome);
  }

  const form = h('form', { class: 'chat-form mt8' },
    h('div', { class: 'col gap4' }, input, h('div', { class: 'spread' }, chipsContainer, send)));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || pending) return;

    pending = true;
    send.disabled = true;
    input.readOnly = true;

    const userBubble = messageBubble('You', text);
    transcript.appendChild(userBubble);
    transcript.scrollTop = transcript.scrollHeight;
    input.value = '';

    status.textContent = 'Co-Pilot is reasoning & executing tools…';
    const turn = { role: 'user', text };

    try {
      const reply = await api('/chat', {
        method: 'POST',
        body: {
          messages: [...messages.slice(-8), turn],
          persona,
          active_mesh_id: activeMeshId || state.currentAssetId || null,
        },
      });

      if (!active) return;
      messages.push(turn, { role: 'model', text: reply.text });

      const agentBubble = messageBubble(persona === 'academic' ? 'Professor 3D' : 'Studio Director', reply.text, reply.tool_call);
      transcript.appendChild(agentBubble);
      transcript.scrollTop = transcript.scrollHeight;
      status.textContent = 'Co-Pilot ready';

      if (reply.tool_call) {
        toast('Autonomous tool executed successfully', 'ok');
        onToolExecuted?.(reply.tool_call);
      }
    } catch (err) {
      if (active) {
        status.textContent = `Error: ${err.message}`;
        transcript.appendChild(h('div', { class: 'form-error' }, err.message));
      }
    } finally {
      pending = false;
      send.disabled = input.readOnly = false;
      input.focus();
    }
  });

  const card = h('div', { class: `chat-dock ${embedded ? 'embedded' : 'card'}` },
    h('div', { class: 'spread mb8' }, personaSeg, reset),
    status,
    transcript,
    form);

  renderChips();
  renderWelcome();

  if (embedded) {
    mount(container, card);
  } else {
    mount(container,
      h('header', { class: 'studio-title' },
        h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
        h('h1', {}, 'Agentic AI 3D Co-Pilot'),
        h('p', { class: 'muted mb0' }, 'Interactive Multimodal Assistant with Autonomous Tool Calling & Academic Tutoring')),
      card);
  }
}

function messageBubble(author, text, toolCall = null) {
  const isUser = author === 'You';
  const bubble = h('article', { class: `chat-message ${isUser ? 'chat-user' : 'chat-agent'}` },
    h('div', { class: 'chat-author mb4' },
      h('strong', { style: isUser ? 'color:var(--accent)' : 'color:var(--cyan)' }, author)),
    h('div', { class: 'chat-content' }, formatMarkdownText(text)));

  if (toolCall) {
    const toolBadge = h('div', { class: 'tool-execution-card mt8 p8' },
      h('div', { class: 'spread' },
        h('span', { class: 'badge completed' }, `⚡ Tool: ${toolCall.action}`),
        h('small', { class: 'mono' }, 'Status: Succeeded')),
      h('p', { class: 'mt4 mb0', style: 'font-size:12.5px;color:var(--text-2);' }, toolCall.message || JSON.stringify(toolCall)));
    bubble.appendChild(toolBadge);
  }

  return bubble;
}

function formatMarkdownText(raw) {
  const container = h('div');
  // Simple clean markdown rendering for bold, lists, and code blocks
  const lines = raw.split('\n');
  lines.forEach(line => {
    if (line.startsWith('### ')) {
      container.appendChild(h('h4', { class: 'mt8 mb4', style: 'color:var(--text);font-size:14px;' }, line.slice(4)));
    } else if (line.startsWith('- ')) {
      container.appendChild(h('li', { style: 'margin-left:16px;font-size:13px;line-height:1.5;' }, line.slice(2)));
    } else if (line.trim().length > 0) {
      container.appendChild(h('p', { class: 'mb4', style: 'font-size:13px;line-height:1.5;' }, line));
    }
  });
  return container;
}
