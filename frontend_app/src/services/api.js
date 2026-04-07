/**
 * API Service - AI Photobooth Kota Madiun
 * Single source of truth for backend communication
 */

const API_BASE = 'http://127.0.0.1:8001/api';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...options.headers },
    ...options,
  };

  const response = await fetch(url, config);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.message || data.detail || `HTTP ${response.status}`);
  }
  return data;
}

export const api = {
  // Health
  health: () => request('/health'),

  // Sessions
  createSession: async () => {
    const res = await request('/photobooth/session', { method: 'POST' });
    return res.data;
  },

  getSession: async (sessionId) => {
    const res = await request(`/photobooth/session/${sessionId}`);
    return res.data;
  },

  // Photos
  uploadPhoto: async (sessionId, dataUrl) => {
    const [header, b64] = dataUrl.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const blob = new Blob([arr], { type: mime });
    const form = new FormData();
    form.append('session_id', sessionId);
    form.append('photo', blob, 'photo.png');

    const response = await fetch(`${API_BASE}/photobooth/upload`, { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || 'Upload failed');
    return data.data;
  },

  processPhoto: async (photoId, backgroundId, mascotId, filterId = null) => {
    const body = { photo_id: photoId, background_id: backgroundId, mascot_id: mascotId };
    if (filterId) body.filter_id = filterId;
    const res = await request('/photobooth/process', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    return res.data;
  },

  getPhoto: async (photoId) => {
    const res = await request(`/photobooth/photo/${photoId}`);
    return res.data;
  },

  downloadByCode: async (code) => {
    const res = await request(`/photobooth/download/${code}`);
    return res.data;
  },

  // Assets
  getBackgrounds: async () => {
    const res = await request('/backgrounds');
    return res.data;
  },

  getMascots: async () => {
    const res = await request('/mascots');
    return res.data;
  },

  getFilters: async () => {
    const res = await request('/filters');
    return res.data;
  },
};
