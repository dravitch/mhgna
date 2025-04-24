import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';
import { Camera, AlertTriangle, CheckCircle2, Info, Download, HelpCircle } from 'lucide-react';

const TrendsVisualization = () => {
  // Données simulées
  const [priceData, setPriceData] = useState([
    { date: '2025-01-15', BTC: 100, ETH: 100, SOL: 100, AVAX: 100, LINK: 100 },
    { date: '2025-01-22', BTC: 103, ETH: 105, SOL: 108, AVAX: 102, LINK: 104 },
    { date: '2025-01-29', BTC: 107, ETH: 103, SOL: 115, AVAX: 106, LINK: 108 },
    { date: '2025-02-05', BTC: 111, ETH: 108, SOL: 128, AVAX: 112, LINK: 106 },
    { date: '2025-02-12', BTC: 108, ETH: 104, SOL: 125, AVAX: 114, LINK: 103 },
    { date: '2025-02-19', BTC: 112, ETH: 110, SOL: 132, AVAX: 120, LINK: 107 },
    { date: '2025-02-26', BTC: 118, ETH: 115, SOL: 138, AVAX: 124, LINK: 110 },
    { date: '2025-03-05', BTC: 122, ETH: 120, SOL: 145, AVAX: 130, LINK: 115 },
    { date: '2025-03-12', BTC: 128, ETH: 125, SOL: 156, AVAX: 136, LINK: 120 },
    { date: '2025-03-19', BTC: 124, ETH: 122, SOL: 148, AVAX: 132, LINK: 118 },
    { date: '2025-03-26', BTC: 130, ETH: 128, SOL: 162, AVAX: 140, LINK: 124 },
    { date: '2025-04-02', BTC: 135, ETH: 132, SOL: 175, AVAX: 148, LINK: 130 },
    { date: '2025-04-08', BTC: 140, ETH: 138, SOL: 182, AVAX: 154, LINK: 135 },
    { date: '2025-04-15', BTC: 138, ETH: 140, SOL: 178, AVAX: 160, LINK: 142 },
  ]);
  
  const [alerts, setAlerts] = useState([
    { asset: 'BTC', drawdown: true, volatility: false, oversold: false, overbought: false },
    { asset: 'ETH', drawdown: false, volatility: false, oversold: false, overbought: false },
    { asset: 'SOL', drawdown: false, volatility: true, oversold: false, overbought: true },
    { asset: 'AVAX', drawdown: false, volatility: false, oversold: false, overbought: false },
    { asset: 'LINK', drawdown: false, volatility: false, oversold: true, overbought: false },
  ]);
  
  const [metrics, setMetrics] = useState([
    { 
      asset: 'SOL', 
      rank: 1,
      centralite: 0.86, 
      momentum: 0.75, 
      stabilite: 0.62,
      return30d: '+22.8%',
      return90d: '+78.0%'
    },
    { 
      asset: 'AVAX', 
      rank: 2,
      centralite: 0.72, 
      momentum: 0.68, 
      stabilite: 0.70,
      return30d: '+17.4%',
      return90d: '+60.0%'
    },
    { 
      asset: 'LINK', 
      rank: 3,
      centralite: 0.68, 
      momentum: 0.55, 
      stabilite: 0.82,
      return30d: '+12.0%',
      return90d: '+42.0%'
    },
    { 
      asset: 'ETH', 
      rank: 4,
      centralite: 0.75, 
      momentum: 0.48, 
      stabilite: 0.78,
      return30d: '+10.7%',
      return90d: '+40.0%'
    },
    { 
      asset: 'BTC', 
      rank: 5,
      centralite: 0.80, 
      momentum: 0.45, 
      stabilite: 0.85,
      return30d: '+8.2%',
      return90d: '+38.0%'
    },
  ]);
  
  // État pour gérer l'affichage des informations d'aide
  const [showHelp, setShowHelp] = useState(false);
  
  // Fonctions utilitaires
  const getLineColor = (asset) => {
    const colorMap = {
      'BTC': '#F7931A',
      'ETH': '#627EEA',
      'SOL': '#00FFA3',
      'AVAX': '#E84142',
      'LINK': '#2A5ADA',
      'BNB': '#F3BA2F',
      'XRP': '#23292F',
      'ADA': '#0033AD',
      'DOT': '#E6007A',
      'MATIC': '#8247E5'
    };
    
    return colorMap[asset] || '#333333';
  };
  
  const getAlertIcon = (alertType, isActive) => {
    if (!isActive) return null;
    
    switch(alertType) {
      case 'drawdown':
        return <AlertTriangle size={18} className="text-red-500" />;
      case 'volatility':
        return <AlertTriangle size={18} className="text-orange-500" />;
      case 'oversold':
        return <CheckCircle2 size={18} className="text-green-500" />;
      case 'overbought':
        return <AlertTriangle size={18} className="text-purple-500" />;
      default:
        return null;
    }
  };
  
  const formatDate = (date) => {
    const parts = date.split('-');
    return `${parts[2]}/${parts[1]}`;
  };
  
  const captureImage = () => {
    // Simuler une capture d'écran (dans une version réelle, utiliserait html2canvas ou similaire)
    alert('Capture d\'écran sauvegardée: mhgna_trends_2025-04-15.png');
  };
  
  const downloadCSV = () => {
    // Préparer les données pour le CSV
    let csvContent = "date,";
    
    // Ajouter les en-têtes des colonnes
    metrics.forEach(metric => {
      csvContent += `${metric.asset},`;
    });
    csvContent = csvContent.slice(0, -1) + "\n";
    
    // Ajouter les données
    priceData.forEach(row => {
      csvContent += `${row.date},`;
      metrics.forEach(metric => {
        csvContent += `${row[metric.asset]},`;
      });
      csvContent = csvContent.slice(0, -1) + "\n";
    });
    
    // Créer un lien de téléchargement
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'mhgna_trends_2025-04-15.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <div className="flex flex-col h-full w-full bg-gray-50 p-4 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h2 className="text-xl font-bold text-gray-800">Tendances du Marché et Signaux MHGNA</h2>
          <button 
            className="ml-2 text-gray-500 hover:text-blue-600"
            onClick={() => setShowHelp(!showHelp)}
          >
            <HelpCircle size={18} />
          </button>
        </div>
        <div className="flex space-x-2">
          <button 
            className="bg-green-600 hover:bg-green-700 text-white rounded-md px-3 py-1 text-sm flex items-center gap-1"
            onClick={downloadCSV}
          >
            <Download size={16} /> Exporter CSV
          </button>
          <button 
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3 py-1 text-sm flex items-center gap-1"
            onClick={captureImage}
          >
            <Camera size={16} /> Exporter Image
          </button>
        </div>
      </div>
      
      {/* Panneau d'aide */}
      {showHelp && (
        <div className="mb-4 bg-blue-50 p-3 rounded-lg border border-blue-200">
          <div className="flex justify-between items-start">
            <h3 className="font-medium text-blue-800 mb-2">Guide d'interprétation</h3>
            <button className="text-blue-500" onClick={() => setShowHelp(false)}>✕</button>
          </div>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Le <span className="font-medium">graphique principal</span> montre l'évolution des prix normalisés à 100 pour faciliter la comparaison.</li>
            <li>• Les <span className="font-medium">lignes plus épaisses</span> représentent les actifs les mieux classés selon l'analyse MHGNA.</li>
            <li>• Le <span className="font-medium">tableau d'alertes</span> montre les signaux actifs pour chaque actif (⚠️ pour les risques, ✓ pour les opportunités).</li>
            <li>• Les <span className="font-medium">métriques</span> sont normalisées entre 0 et 1 (plus élevé = meilleur).</li>
            <li>• Les <span className="font-medium">zones rouges</span> sur le graphique indiquent des périodes où le Bitcoin était en drawdown significatif.</li>
          </ul>
        </div>
      )}
      
      {/* Graphique des prix */}
      <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <h3 className="text-lg font-medium text-gray-700 mb-3">Évolution des prix sur 90 jours (normalisée à 100)</h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={priceData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis 
                dataKey="date" 
                tickFormatter={formatDate}
                padding={{ left: 10, right: 10 }}
              />
              <YAxis domain={['dataMin - 5', 'dataMax + 5']}>
                <Label
                  value="Prix normalisé"
                  angle={-90}
                  position="insideLeft"
                  style={{ textAnchor: 'middle', fill: '#666' }}
                />
              </YAxis>
              <Tooltip 
                formatter={(value, name) => [`${value}`, name]}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Legend verticalAlign="top" height={36} />
              
              {metrics.map((metric) => (
                <Line
                  key={metric.asset}
                  type="monotone"
                  dataKey={metric.asset}
                  stroke={getLineColor(metric.asset)}
                  strokeWidth={metric.rank <= 3 ? 3 : 2}
                  dot={false}
                  activeDot={{ r: 8 }}
                  name={`${metric.asset} (Rang ${metric.rank})`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Tableau des signaux d'alerte */}
      <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <h3 className="text-lg font-medium text-gray-700 mb-3">Signaux d'alerte actifs</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-50">
              <tr>
                <th className="py-2 px-4 border-b text-left">Actif</th>
                <th className="py-2 px-4 border-b text-center">Drawdown</th>
                <th className="py-2 px-4 border-b text-center">Volatilité</th>
                <th className="py-2 px-4 border-b text-center">Survendu<br/>(opportunité)</th>
                <th className="py-2 px-4 border-b text-center">Suracheté<br/>(risque)</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert, index) => (
                <tr key={alert.asset} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-2 px-4 border-b font-medium" style={{ color: getLineColor(alert.asset) }}>
                    {alert.asset}
                  </td>
                  <td className="py-2 px-4 border-b text-center">
                    {getAlertIcon('drawdown', alert.drawdown)}
                  </td>
                  <td className="py-2 px-4 border-b text-center">
                    {getAlertIcon('volatility', alert.volatility)}
                  </td>
                  <td className="py-2 px-4 border-b text-center">
                    {getAlertIcon('oversold', alert.oversold)}
                  </td>
                  <td className="py-2 px-4 border-b text-center">
                    {getAlertIcon('overbought', alert.overbought)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Métriques des actifs recommandés */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-700 mb-3">Métriques des actifs recommandés</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead className="bg-gray-50">
                <tr>
                  <th className="py-2 px-3 border-b text-left">Rang</th>
                  <th className="py-2 px-3 border-b text-left">Actif</th>
                  <th className="py-2 px-3 border-b text-center">Centralité</th>
                  <th className="py-2 px-3 border-b text-center">Momentum</th>
                  <th className="py-2 px-3 border-b text-center">Stabilité</th>
                </tr>
              </thead>
              <tbody>
                {metrics.filter(m => m.rank <= 3).map((metric, index) => (
                  <tr key={metric.asset} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="py-2 px-3 border-b font-bold">{metric.rank}</td>
                    <td className="py-2 px-3 border-b font-medium" style={{ color: getLineColor(metric.asset) }}>
                      {metric.asset}
                    </td>
                    <td className="py-2 px-3 border-b text-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${metric.centralite * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600">{metric.centralite.toFixed(2)}</span>
                    </td>
                    <td className="py-2 px-3 border-b text-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-green-500 h-2.5 rounded-full" 
                          style={{ width: `${metric.momentum * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600">{metric.momentum.toFixed(2)}</span>
                    </td>
                    <td className="py-2 px-3 border-b text-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-purple-500 h-2.5 rounded-full" 
                          style={{ width: `${metric.stabilite * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600">{metric.stabilite.toFixed(2)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-700 mb-3">Rendements récents</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead className="bg-gray-50">
                <tr>
                  <th className="py-2 px-3 border-b text-left">Rang</th>
                  <th className="py-2 px-3 border-b text-left">Actif</th>
                  <th className="py-2 px-3 border-b text-center">Rend. 30j</th>
                  <th className="py-2 px-3 border-b text-center">Rend. 90j</th>
                </tr>
              </thead>
              <tbody>
                {metrics.filter(m => m.rank <= 3).map((metric, index) => (
                  <tr key={metric.asset} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="py-2 px-3 border-b font-bold">{metric.rank}</td>
                    <td className="py-2 px-3 border-b font-medium" style={{ color: getLineColor(metric.asset) }}>
                      {metric.asset}
                    </td>
                    <td className="py-2 px-3 border-b text-center text-green-600 font-medium">
                      {metric.return30d}
                    </td>
                    <td className="py-2 px-3 border-b text-center text-green-600 font-medium">
                      {metric.return90d}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <div className="mt-4 bg-white p-3 rounded-lg border border-gray-200">
        <div className="flex items-start gap-2">
          <Info size={18} className="text-blue-500 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-gray-700">Actions recommandées :</h3>
            <ul className="text-sm text-gray-600 space-y-1 mt-1">
              <li>• Considérer une position sur : <span className="font-medium">SOL, AVAX, LINK</span></li>
              <li>• Prudence conseillée sur <span className="font-medium">SOL</span> en raison de sa volatilité élevée et de signes de surachat</li>
              <li>• Opportunité potentielle sur <span className="font-medium">LINK</span> qui montre des signes de survente</li>
              <li>• Vigilance sur <span className="font-medium">BTC</span> qui présente une alerte de drawdown</li>
            </ul>
          </div>
        </div>
      </div>
      
      {/* Footer avec la date de mise à jour */}
      <div className="mt-4 text-xs text-gray-500 text-right">
        <p>Données mises à jour le: 15/04/2025 • Analyse basée sur MHGNA v1.2.0</p>
      </div>
    </div>
  );
};

export default TrendsVisualization;