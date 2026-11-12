import { api, h, mount, state } from './lib.js';

const QUESTIONS = [
  'What makes a good reference image?',
  'How do TripoSR, Depth Anything V2 and MiDaS compare?',
  'Which format should I download for Blender?',
];

export async function renderChat(_params, main) {
  const messages = [];
  let active = true;
  let pending = false;
  state.routeCleanup = () => { active = false; };
  const status = h('p', { class: 'muted', role: 'status' }, 'Checking Gemini setup…');
  const transcript = h('div', { class: 'chat-transcript', role: 'log', 'aria-label': 'Conversation' });
  const input = h('textarea', { id: 'chat-message', rows: '2', maxlength: '2000', required: true,
    disabled: true, placeholder: 'Ask about photos, meshes, or exports' });
  const send = h('button', { class: 'btn primary', type: 'submit', disabled: true }, 'Send');
  const check = h('button', { class: 'btn sm chat-setup', type: 'button', hidden: true, onclick: checkSetup }, 'Check setup');
  const reset = h('button', { class: 'btn sm', type: 'button', disabled: true, onclick: () => {
    messages.length = 0;
    input.value = '';
    mount(transcript);
    reset.disabled = true;
    status.textContent = 'Ready to chat';
    input.focus();
  } }, 'Clear chat');
  const suggestions = QUESTIONS.map(question => h('button', { class: 'btn sm', type: 'button', disabled: true,
    onclick: () => { input.value = question; input.focus(); } }, question));
  const form = h('form', { class: 'chat-form' }, h('label', { class: 'field', for: 'chat-message' },
    h('span', {}, 'Message'), input), send);

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || pending) return;
    pending = true;
    reset.disabled = send.disabled = true;
    input.readOnly = true;
    suggestions.forEach(button => { button.disabled = true; });
    status.textContent = 'Gemini is replying…';
    const turn = { role: 'user', text };
    const bubble = messageBubble('You', text);
    transcript.append(bubble);
    transcript.scrollTop = transcript.scrollHeight;
    try {
      const reply = await api('/chat', { method: 'POST', body: { messages: [...messages.slice(-10), turn] } });
      if (!active) return;
      messages.push(turn, { role: 'model', text: reply.text.slice(0, 4000) });
      transcript.append(messageBubble('Gemini', reply.text));
      transcript.scrollTop = transcript.scrollHeight;
      status.textContent = 'Ready to chat';
      input.value = '';
    } catch (err) {
      bubble.remove();
      if (active) status.textContent = err.message;
    } finally {
      pending = false;
      if (active) {
        send.disabled = input.readOnly = false;
        reset.disabled = messages.length === 0;
        suggestions.forEach(button => { button.disabled = false; });
        input.focus();
      }
    }
  });

  mount(main, h('header', { class: 'studio-title' }, h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
    h('h1', {}, 'Gemini chat'), h('p', { class: 'mb0' }, 'Ask about image preparation, 3D reconstruction, and model formats.')),
  h('section', { class: 'card chat-card' }, h('div', { class: 'spread' }, h('h2', { class: 'mb0' }, 'Study assistant'), reset),
    status, check, h('div', { class: 'row wrap gap8' }, suggestions), transcript, form,
    h('p', { class: 'hint mb0' }, 'Chat messages are sent to Google. Image uploads and 3D generation run locally.')));

  async function checkSetup() {
    check.disabled = true;
    status.textContent = 'Checking Gemini setup…';
    try {
      const config = await api('/chat/status');
      if (!active) return;
      check.hidden = config.configured;
      send.disabled = input.disabled = !config.configured;
      suggestions.forEach(button => { button.disabled = !config.configured; });
      if (config.configured) {
        status.textContent = `Key loaded · ${config.model}`;
      } else {
        mount(status, 'Set ', h('code', {}, 'GEMINI_API_KEY'), ' in the project’s ', h('code', {}, '.env'),
          ' file, then select Check setup. ', h('a', { href: 'https://aistudio.google.com/apikey', target: '_blank',
            rel: 'noopener noreferrer' }, 'Get a Gemini API key'));
      }
    } catch (err) {
      if (active) { status.textContent = err.message; check.hidden = false; }
    } finally {
      check.disabled = false;
    }
  }
  await checkSetup();
}

function messageBubble(author, text) {
  return h('article', { class: `chat-message ${author === 'You' ? 'chat-user' : ''}` },
    h('strong', {}, author), h('p', { class: 'mb0' }, text));
}
