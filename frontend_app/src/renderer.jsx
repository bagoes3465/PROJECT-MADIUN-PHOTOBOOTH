import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import Home from './pages/Home';
import Camera from './pages/Camera';
import Customize from './pages/Customize';
import Processing from './pages/Processing';
import Result from './pages/Result';
import ErrorPage from './pages/Error';
import { api } from './services/api';

function App() {
  const [page, setPage] = useState('home');
  const [sessionData, setSessionData] = useState(null);
  const [photoData, setPhotoData] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [customization, setCustomization] = useState(null);
  const [resultData, setResultData] = useState(null);
  const [errorInfo, setErrorInfo] = useState({ message: '', detail: '' });

  const resetAll = () => {
    setPage('home');
    setSessionData(null);
    setPhotoData(null);
    setCapturedImage(null);
    setCustomization(null);
    setResultData(null);
    setErrorInfo({ message: '', detail: '' });
  };

  const showError = (message, detail = '') => {
    setErrorInfo({ message, detail });
    setPage('error');
  };

  // Home → Camera: create session first
  const handleStart = async () => {
    try {
      const session = await api.createSession();
      setSessionData(session);
      setPage('camera');
    } catch (err) {
      showError('Gagal membuat sesi', err.message);
    }
  };

  // Camera → Uploading → Customize: upload photo
  const handleCaptured = async (dataUrl) => {
    try {
      if (!sessionData?.session_id) {
        showError('Sesi tidak valid', 'Silakan mulai ulang.');
        return;
      }
      setCapturedImage(dataUrl);
      setPage('uploading');
      const photo = await api.uploadPhoto(sessionData.session_id, dataUrl);
      setPhotoData(photo);
      setPage('customize');
    } catch (err) {
      showError('Gagal mengunggah foto', err.message);
    }
  };

  // Customize → Processing: save customization
  const handleCustomize = (data) => {
    setCustomization(data);
    setPage('processing');
  };

  // Processing: call process API
  const handleProcess = async () => {
    try {
      const result = await api.processPhoto(
        photoData.photo_id,
        customization.backgroundId,
        customization.mascotId,
        customization.filterId,
      );
      setResultData(result);
      setPage('result');
    } catch (err) {
      showError('Gagal memproses foto', err.message);
    }
  };

  return (
    <>
      {page === 'home' && <Home onStart={handleStart} />}
      {page === 'camera' && (
        <Camera onBack={resetAll} onCaptured={handleCaptured} />
      )}
      {page === 'uploading' && (
        <div className="pb-page">
          <div className="pb-uploading-container">
            <img src={capturedImage} alt="Captured" className="pb-uploading-preview" />
            <div className="pb-uploading-overlay">
              <div className="pb-uploading-spinner" />
              <p className="pb-uploading-text">Mengunggah foto...</p>
            </div>
          </div>
        </div>
      )}
      {page === 'customize' && (
        <Customize
          capturedImage={capturedImage}
          onBack={() => setPage('camera')}
          onNext={handleCustomize}
        />
      )}
      {page === 'processing' && <Processing onProcess={handleProcess} />}
      {page === 'result' && (
        <Result result={resultData} onHome={resetAll} />
      )}
      {page === 'error' && (
        <ErrorPage
          message={errorInfo.message}
          detail={errorInfo.detail}
          onRetry={() => setPage('camera')}
          onHome={resetAll}
        />
      )}
    </>
  );
}

const root = createRoot(document.getElementById('root'));
root.render(<App />);
