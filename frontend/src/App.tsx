import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';
import { Overview } from './pages/Overview';
import { PredictionStudio } from './pages/PredictionStudio';
import { ModelEvaluation } from './pages/ModelEvaluation';
import { DatasetExplorer } from './pages/DatasetExplorer';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          {/* Strictly Three Primary Pages */}
          <Route index element={<Overview />} />
          <Route path="predict" element={<PredictionStudio />} />
          <Route path="evaluation" element={<ModelEvaluation />} />

          {/* Sub-routes & Aliases */}
          <Route path="gameplay" element={<Navigate to="/predict" replace />} />
          <Route path="dataset" element={<DatasetExplorer />} />
          <Route path="experiments" element={<Navigate to="/evaluation" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
