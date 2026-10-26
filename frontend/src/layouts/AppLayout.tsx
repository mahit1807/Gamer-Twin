import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  BarChart3
} from 'lucide-react';
import { api, SystemStatus } from '../services/api';

const NAV_TABS = [
  { path: '/', label: 'Overview', subtitle: 'Project & Pipeline', icon: LayoutDashboard },
  { path: '/predict', label: 'Next Prediction', subtitle: 'Inference & What-If', icon: Zap },
  { path: '/evaluation', label: 'Model & Evaluation', subtitle: 'Real Benchmarks', icon: BarChart3 },
];

export const AppLayout: React.FC = () => {
  const [, setSystemStatus] = useState<SystemStatus | null>(null);
  const [, setBackendOnline] = useState<boolean | null>(null);
  const location = useLocation();

  useEffect(() => {
    const checkStatus = () => {
      api.getSystemStatus()
        .then((data) => {
          setSystemStatus(data);
          setBackendOnline(true);
        })
        .catch(() => {
          setBackendOnline(false);
        });
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#F8F5EF] text-[#302721] font-sans">
      {/* Top Academic Navigation Bar */}
      <header className="sticky top-0 z-50 bg-white border-b border-[#EFE7DB] shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Wordmark & Identity */}
            <div className="flex items-center space-x-3">
              <div className="h-9 w-9 rounded-lg bg-[#C66A4E] text-white flex items-center justify-center font-bold text-sm tracking-tight shadow-xs">
                GT
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-base tracking-tight text-[#302721]">GAMER-TWIN</span>
                </div>
                <div className="text-[11px] text-[#81786E]">
                  Symbiosis Institute of Technology, Pune · AIML
                </div>
              </div>
            </div>

            {/* Exactly 3 Primary Navigation Tabs */}
            <nav className="hidden md:flex items-center space-x-1">
              {NAV_TABS.map((tab) => {
                const Icon = tab.icon;
                const isActive = location.pathname === tab.path;
                return (
                  <NavLink
                    key={tab.path}
                    to={tab.path}
                    className={`flex items-center px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
                      isActive
                        ? 'bg-[#C66A4E] text-white shadow-xs'
                        : 'text-[#81786E] hover:text-[#302721] hover:bg-[#F8F5EF]'
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 mr-2 ${isActive ? 'text-white' : 'text-[#81786E]'}`} />
                    <span>{tab.label}</span>
                  </NavLink>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Mobile Navigation Row */}
        <div className="md:hidden border-t border-[#EFE7DB] bg-white px-2 py-1.5 flex justify-around">
          {NAV_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = location.pathname === tab.path;
            return (
              <NavLink
                key={tab.path}
                to={tab.path}
                className={`flex flex-col items-center py-1 px-2 text-[10px] font-medium rounded ${
                  isActive ? 'text-[#C66A4E] font-bold' : 'text-[#81786E]'
                }`}
              >
                <Icon className="w-4 h-4 mb-0.5" />
                {tab.label.split(' ')[0]}
              </NavLink>
            );
          })}
        </div>
      </header>

      {/* Main Page Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Shared Academic Presentation Footer */}
      <footer className="bg-white border-t border-[#EFE7DB] py-5 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-[#81786E] gap-3">
          <div>
            <strong>GAMER-TWIN</strong> — Final-Semester Undergraduate AI & ML Project Presentation.
          </div>
          <div className="flex items-center space-x-4">
            <span>Students: <strong>Mahit Turakhia</strong> & <strong>Het Shah</strong></span>
            <span>·</span>
            <span>Guide: <strong>Dr. Ananya Bhattacharjee</strong></span>
          </div>
        </div>
      </footer>
    </div>
  );
};
