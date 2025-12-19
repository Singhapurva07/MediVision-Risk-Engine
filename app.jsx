import { useState } from "react";

const Activity = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
);

const AlertCircle = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
);

const ChevronDown = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9"/></svg>
);

const Pill = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.5 20.5 21 10a7 7 0 1 0-10-10l-10.5 10.5a7 7 0 1 0 10 10Z"/><path d="m8.5 8.5 7 7"/></svg>
);

const AlertTriangle = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
);

const Shield = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
);

export default function App() {
  const [input, setInput] = useState({});
  const [result, setResult] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const requiredFields = [
    { key: "age", label: "Age", type: "number", placeholder: "e.g., 45" },
    { 
      key: "gender", 
      label: "Gender", 
      type: "select",
      options: [
        { value: "", label: "Select Gender" },
        { value: "M", label: "Male" },
        { value: "F", label: "Female" }
      ]
    },
    { 
      key: "drug_class", 
      label: "Drug Class", 
      type: "select",
      options: [
        { value: "", label: "Select Drug Class" },
        { value: "Opioid", label: "Opioid" },
        { value: "Stimulant", label: "Stimulant" },
        { value: "Sedative", label: "Sedative" },
        { value: "Benzodiazepine", label: "Benzodiazepine" }
      ]
    },
    { key: "drug_potency", label: "Drug Potency", type: "number", placeholder: "0-10" },
    { key: "daily_dose", label: "Daily Dose (mg)", type: "number", placeholder: "e.g., 50" },
    { key: "refill_count", label: "Refill Count", type: "number", placeholder: "e.g., 3" }
  ];

  const advancedFields = [
    { 
      key: "income_class", 
      label: "Income Class", 
      type: "select",
      options: [
        { value: "", label: "Select Income Class" },
        { value: "low", label: "Low" },
        { value: "middle", label: "Middle" },
        { value: "high", label: "High" }
      ]
    },
    { 
      key: "urban_flag", 
      label: "Urban Area", 
      type: "select",
      options: [
        { value: "", label: "Select" },
        { value: "1", label: "Yes" },
        { value: "0", label: "No" }
      ]
    },
    { key: "max_safe_dose", label: "Max Safe Dose (mg)", type: "number", placeholder: "e.g., 100" },
    { key: "early_refill", label: "Early Refills", type: "number", placeholder: "e.g., 1" },
    { key: "dose_escalation_rate", label: "Dose Escalation Rate", type: "number", placeholder: "0-1" },
    { key: "number_of_doctors", label: "Number of Doctors", type: "number", placeholder: "e.g., 2" },
    { key: "number_of_pharmacies", label: "Number of Pharmacies", type: "number", placeholder: "e.g., 1" },
    { key: "anxiety", label: "Anxiety Level", type: "number", placeholder: "0-10" },
    { key: "depression", label: "Depression Level", type: "number", placeholder: "0-10" },
    { key: "stress", label: "Stress Level", type: "number", placeholder: "0-10" },
    { key: "compulsive_use", label: "Compulsive Use", type: "number", placeholder: "0-10" },
    { key: "missed_doses", label: "Missed Doses", type: "number", placeholder: "e.g., 2" },
    { key: "overuse", label: "Overuse Incidents", type: "number", placeholder: "e.g., 1" },
    { key: "liver_score", label: "Liver Function Score", type: "number", placeholder: "0-100" },
    { key: "kidney_score", label: "Kidney Function Score", type: "number", placeholder: "0-100" },
    { key: "blood_pressure", label: "Blood Pressure (mmHg)", type: "number", placeholder: "e.g., 120" },
    { key: "heart_rate", label: "Heart Rate (bpm)", type: "number", placeholder: "e.g., 72" }
  ];

  const analyzeRiskFactors = (inputData, score) => {
    const factors = { protective: [], concerning: [], critical: [] };

    if (inputData.drug_potency > 7) {
      factors.critical.push({ text: "High drug potency increases addiction vulnerability", value: `Potency: ${inputData.drug_potency}/10` });
    } else if (inputData.drug_potency > 4) {
      factors.concerning.push({ text: "Moderate drug potency requires monitoring", value: `Potency: ${inputData.drug_potency}/10` });
    } else if (inputData.drug_potency) {
      factors.protective.push({ text: "Lower potency medication reduces risk", value: `Potency: ${inputData.drug_potency}/10` });
    }

    if (inputData.daily_dose && inputData.max_safe_dose) {
      const ratio = inputData.daily_dose / inputData.max_safe_dose;
      if (ratio > 0.9) {
        factors.critical.push({ text: "Daily dose approaching maximum safe limit", value: `${inputData.daily_dose}mg / ${inputData.max_safe_dose}mg max` });
      } else if (ratio > 0.6) {
        factors.concerning.push({ text: "Elevated dosage level requires attention", value: `${inputData.daily_dose}mg / ${inputData.max_safe_dose}mg max` });
      } else {
        factors.protective.push({ text: "Dosage within safe therapeutic range", value: `${inputData.daily_dose}mg / ${inputData.max_safe_dose}mg max` });
      }
    }

    if (inputData.early_refill > 2) {
      factors.critical.push({ text: "Frequent early refills suggest problematic use", value: `${inputData.early_refill} early refills` });
    } else if (inputData.early_refill > 0) {
      factors.concerning.push({ text: "Some early refills detected", value: `${inputData.early_refill} early refills` });
    }

    if (inputData.number_of_doctors > 2) {
      factors.critical.push({ text: "Multiple prescribers may indicate doctor shopping", value: `${inputData.number_of_doctors} different doctors` });
    }

    if (inputData.number_of_pharmacies > 2) {
      factors.concerning.push({ text: "Using multiple pharmacies", value: `${inputData.number_of_pharmacies} different pharmacies` });
    }

    const mentalHealthFactors = [];
    if (inputData.anxiety > 7) mentalHealthFactors.push(`high anxiety (${inputData.anxiety}/10)`);
    if (inputData.depression > 7) mentalHealthFactors.push(`high depression (${inputData.depression}/10)`);
    if (inputData.stress > 7) mentalHealthFactors.push(`high stress (${inputData.stress}/10)`);

    if (mentalHealthFactors.length > 0) {
      factors.concerning.push({ text: "Elevated mental health symptoms increase vulnerability", value: mentalHealthFactors.join(", ") });
    }

    if (inputData.compulsive_use > 7) {
      factors.critical.push({ text: "High compulsive use patterns detected", value: `Compulsivity: ${inputData.compulsive_use}/10` });
    }

    if (inputData.overuse > 2) {
      factors.critical.push({ text: "Multiple overuse incidents", value: `${inputData.overuse} incidents` });
    }

    if (inputData.liver_score < 60) {
      factors.concerning.push({ text: "Liver function below optimal range", value: `Score: ${inputData.liver_score}/100` });
    } else if (inputData.liver_score >= 80) {
      factors.protective.push({ text: "Good liver function", value: `Score: ${inputData.liver_score}/100` });
    }

    if (inputData.kidney_score < 60) {
      factors.concerning.push({ text: "Kidney function below optimal range", value: `Score: ${inputData.kidney_score}/100` });
    } else if (inputData.kidney_score >= 80) {
      factors.protective.push({ text: "Good kidney function", value: `Score: ${inputData.kidney_score}/100` });
    }

    if (inputData.drug_class === "Opioid" || inputData.drug_class === "Benzodiazepine") {
      factors.concerning.push({ text: `${inputData.drug_class} medications carry higher addiction potential`, value: `${inputData.drug_class} class` });
    }

    return factors;
  };

  const getClinicalRecommendations = (score, inputData, riskAnalysis) => {
    const recommendations = {
      immediate: [],
      monitoring: [],
      treatment: [],
      lifestyle: [],
      support: []
    };

    // Base recommendations on risk level
    if (score < 30) {
      recommendations.immediate.push("Continue current treatment regimen as prescribed");
      recommendations.monitoring.push("Schedule routine follow-up appointments every 3-6 months");
      recommendations.treatment.push("Maintain current therapeutic approach");
      recommendations.lifestyle.push("Continue healthy lifestyle habits and stress management");
      recommendations.support.push("Maintain open communication with healthcare provider");
    } else if (score < 60) {
      recommendations.immediate.push("Schedule follow-up appointment within 2 weeks");
      recommendations.monitoring.push("Increase monitoring frequency to monthly appointments");
      recommendations.treatment.push("Review and potentially adjust current medication regimen");
      recommendations.lifestyle.push("Develop comprehensive stress management plan");
      recommendations.support.push("Consider joining structured support groups (NA, SMART Recovery)");
    } else {
      recommendations.immediate.push("URGENT: Schedule immediate consultation with addiction specialist");
      recommendations.immediate.push("Implement immediate medication safety measures (secure storage, supervision)");
      recommendations.monitoring.push("Daily to weekly monitoring by healthcare team required");
      recommendations.treatment.push("Urgent referral to addiction medicine specialist");
      recommendations.lifestyle.push("Complete removal of unsecured medications from home environment");
      recommendations.support.push("Mandatory participation in intensive recovery programs");
    }

    // Personalized recommendations based on specific risk factors
    
    // High potency medications
    if (inputData.drug_potency > 7) {
      recommendations.treatment.push(`Address high-potency ${inputData.drug_class} use - evaluate dose reduction or alternative medications`);
      recommendations.monitoring.push("Closely monitor for signs of tolerance and dose escalation patterns");
    }

    // Dosage concerns
    if (inputData.daily_dose && inputData.max_safe_dose) {
      const ratio = inputData.daily_dose / inputData.max_safe_dose;
      if (ratio > 0.9) {
        recommendations.immediate.push("Daily dose near maximum limit - immediate prescriber consultation required");
        recommendations.treatment.push("Evaluate urgent dose reduction strategy or medication rotation");
      } else if (ratio > 0.6) {
        recommendations.monitoring.push("Track dosage levels closely - consider dose optimization");
      }
    }

    // Early refills
    if (inputData.early_refill > 2) {
      recommendations.immediate.push("Multiple early refills detected - implement medication tracking system immediately");
      recommendations.monitoring.push("Implement prescription drug monitoring program (PDMP) checks at every refill");
      recommendations.treatment.push("Establish structured medication agreement with clear refill policies");
    } else if (inputData.early_refill > 0) {
      recommendations.monitoring.push("Monitor refill patterns for early request trends");
    }

    // Multiple prescribers
    if (inputData.number_of_doctors > 2) {
      recommendations.immediate.push("Multiple prescribers identified - coordinate care between all providers immediately");
      recommendations.monitoring.push("Establish single point of care coordination for all prescriptions");
      recommendations.treatment.push("Centralize prescribing through primary addiction specialist or pain management physician");
    }

    // Multiple pharmacies
    if (inputData.number_of_pharmacies > 2) {
      recommendations.immediate.push("Consolidate all prescriptions to single pharmacy for better monitoring");
      recommendations.monitoring.push("Implement pharmacy-based medication therapy management");
    }

    // Mental health factors
    const mentalHealthIssues = [];
    if (inputData.anxiety > 7) mentalHealthIssues.push("anxiety");
    if (inputData.depression > 7) mentalHealthIssues.push("depression");
    if (inputData.stress > 7) mentalHealthIssues.push("stress");

    if (mentalHealthIssues.length > 0) {
      recommendations.treatment.push(`Address elevated ${mentalHealthIssues.join(", ")} through integrated mental health treatment`);
      recommendations.treatment.push("Consider cognitive behavioral therapy (CBT) or dialectical behavior therapy (DBT)");
      recommendations.lifestyle.push(`Implement targeted interventions for ${mentalHealthIssues.join(", ")} management`);
      recommendations.support.push("Connect with mental health support groups and counseling services");
    }

    // Compulsive use
    if (inputData.compulsive_use > 7) {
      recommendations.treatment.push("High compulsivity detected - evaluate for obsessive-compulsive spectrum disorder");
      recommendations.treatment.push("Consider specialized behavioral interventions for compulsive medication use");
      recommendations.monitoring.push("Track compulsive use patterns and triggers in daily logs");
    }

    // Overuse incidents
    if (inputData.overuse > 2) {
      recommendations.immediate.push("Multiple overuse incidents - implement immediate harm reduction strategies");
      recommendations.treatment.push("Evaluate need for medically supervised detoxification or stabilization");
      recommendations.monitoring.push("Consider directly observed therapy or supervised medication administration");
    } else if (inputData.overuse > 0) {
      recommendations.monitoring.push("Monitor for additional overuse incidents - educate on proper dosing");
    }

    // Liver function
    if (inputData.liver_score < 60) {
      recommendations.monitoring.push("Monitor liver function tests every 3 months due to reduced hepatic function");
      recommendations.treatment.push("Consider hepatoprotective measures and dose adjustments for liver impairment");
      recommendations.lifestyle.push("Eliminate alcohol consumption and hepatotoxic substances");
    }

    // Kidney function
    if (inputData.kidney_score < 60) {
      recommendations.monitoring.push("Monitor kidney function and medication levels due to renal impairment");
      recommendations.treatment.push("Adjust medication dosing based on creatinine clearance and renal function");
      recommendations.lifestyle.push("Maintain adequate hydration and follow renal-protective diet");
    }

    // Drug class specific
    if (inputData.drug_class === "Opioid") {
      recommendations.treatment.push("For opioid use: Consider medication-assisted treatment (buprenorphine, naltrexone, methadone)");
      recommendations.immediate.push("Keep naloxone (Narcan) readily available for overdose prevention");
      recommendations.monitoring.push("Screen for opioid use disorder using validated assessment tools");
    } else if (inputData.drug_class === "Benzodiazepine") {
      recommendations.treatment.push("For benzodiazepine use: Evaluate gradual tapering plan to avoid withdrawal seizures");
      recommendations.monitoring.push("Monitor for paradoxical reactions and cognitive impairment");
      recommendations.lifestyle.push("Avoid alcohol and other CNS depressants completely");
    } else if (inputData.drug_class === "Stimulant") {
      recommendations.monitoring.push("Monitor cardiovascular function including blood pressure and heart rate");
      recommendations.treatment.push("Screen for cardiovascular complications of stimulant use");
    }

    // Cardiovascular monitoring if data available
    if (inputData.blood_pressure > 140 || inputData.heart_rate > 100) {
      recommendations.monitoring.push("Cardiovascular parameters elevated - monitor blood pressure and heart rate regularly");
      recommendations.treatment.push("Evaluate cardiovascular risk and consider cardiology consultation");
    }

    // Dose escalation
    if (inputData.dose_escalation_rate > 0.5) {
      recommendations.immediate.push("Rapid dose escalation detected - evaluate for tolerance development");
      recommendations.treatment.push("Implement dose stabilization protocol and evaluate alternative treatments");
    }

    // Add crisis resources for moderate to high risk
    if (score >= 30) {
      recommendations.support.push("24/7 Crisis Resources: 988 Suicide & Crisis Lifeline, SAMHSA Helpline (1-800-662-4357)");
    }

    return recommendations;
  };

  const handleChange = (k, v) => {
    setInput(prev => ({ ...prev, [k]: v === "" ? undefined : (isNaN(v) ? v : Number(v)) }));
  };

  const handlePredict = async () => {
    setErr(null);
    setResult(null);
    setAnalysis(null);
    setLoading(true);

    const missingRequired = requiredFields.filter(f => !input[f.key]);
    if (missingRequired.length > 0) {
      setErr(`Please fill in: ${missingRequired.map(f => f.label).join(", ")}`);
      setLoading(false);
      return;
    }

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input)
      });
      const data = await res.json();
      
      if (data.success) {
        setResult(data.risk_score);
        setAnalysis(analyzeRiskFactors(input, data.risk_score));
      } else {
        setErr("Backend rejected input");
      }
    } catch (e) {
      setErr("Prediction failed. Please check your inputs and ensure the server is running.");
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (score) => {
    if (score < 30) return { text: "Low Risk", color: "text-green-500", bg: "bg-green-50", border: "border-green-200" };
    if (score < 60) return { text: "Moderate Risk", color: "text-yellow-600", bg: "bg-yellow-50", border: "border-yellow-200" };
    return { text: "High Risk", color: "text-red-500", bg: "bg-red-50", border: "border-red-200" };
  };

  const renderField = (field) => {
    const isRequired = requiredFields.includes(field);
    
    if (field.type === "select") {
      return (
        <div key={field.key} className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {field.label} {isRequired && <span className="text-red-500">*</span>}
          </label>
          <div className="relative">
            <select
              value={input[field.key] || ""}
              onChange={(e) => handleChange(field.key, e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-300 rounded-lg bg-white text-gray-900 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {field.options.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-3 w-5 h-5 text-gray-400 pointer-events-none" />
          </div>
        </div>
      );
    }

    return (
      <div key={field.key} className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {field.label} {isRequired && <span className="text-red-500">*</span>}
        </label>
        <input
          type={field.type}
          placeholder={field.placeholder}
          value={input[field.key] || ""}
          onChange={(e) => handleChange(field.key, e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <Pill className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              MediVision
            </h1>
          </div>
          <p className="text-gray-600 text-lg">AI-Powered Addiction Risk Assessment Platform</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Left Column - Input Form */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-6 sticky top-6">
              <div className="mb-6">
                <div className="flex items-center gap-2 mb-4">
                  <Activity className="w-5 h-5 text-blue-600" />
                  <h2 className="text-xl font-bold text-gray-800">Patient Information</h2>
                </div>
                
                <div className="space-y-4">
                  {requiredFields.map(renderField)}
                </div>
              </div>

              <div className="border-t pt-6">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-gray-700 font-medium hover:text-blue-600 transition-colors mb-4"
                >
                  <ChevronDown className={`w-5 h-5 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  <span>Advanced Parameters (Optional)</span>
                </button>

                {showAdvanced && (
                  <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                    {advancedFields.map(renderField)}
                  </div>
                )}
              </div>

              <button
                onClick={handlePredict}
                disabled={loading}
                className="w-full mt-6 py-3.5 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Analyzing..." : "Analyze Risk Profile"}
              </button>

              {err && (
                <div className="mt-4 bg-red-50 border-2 border-red-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-red-800 text-sm mb-1">Error</h3>
                      <p className="text-red-600 text-sm">{err}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {result === null ? (
              <div className="bg-white rounded-2xl shadow-xl p-12 text-center">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-400 mb-2">No Analysis Yet</h3>
                <p className="text-gray-500">Fill in the patient information and click "Analyze Risk Profile" to begin assessment</p>
              </div>
            ) : (
              <>
                {/* Risk Score Card */}
                <div className={`${getRiskLevel(result).bg} rounded-2xl shadow-xl p-8 border-2 ${getRiskLevel(result).border}`}>
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-800 mb-4">Risk Assessment Score</h3>
                    <div className="mb-4">
                      <div className={`text-7xl font-bold mb-2 ${getRiskLevel(result).color}`}>
                        {result.toFixed(1)}
                      </div>
                      <div className={`text-2xl font-semibold ${getRiskLevel(result).color}`}>
                        {getRiskLevel(result).text}
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 mt-4">
                      <div 
                        className={`h-3 rounded-full ${result < 30 ? 'bg-green-500' : result < 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.min(result, 100)}%` }}
                      ></div>
                    </div>
                    <p className="text-gray-600 text-sm mt-4">
                      This score is based on multiple clinical risk factors and behavioral patterns.
                    </p>
                  </div>
                </div>

                {/* Risk Analysis */}
                {analysis && (
                  <div className="bg-white rounded-2xl shadow-xl p-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                      <Activity className="w-6 h-6 text-blue-600" />
                      Risk Factor Analysis
                    </h3>

                    {analysis.critical.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-base font-semibold text-red-600 mb-2 flex items-center gap-2">
                          <AlertTriangle className="w-5 h-5" />
                          Critical Risk Factors ({analysis.critical.length})
                        </h4>
                        <div className="space-y-2">
                          {analysis.critical.map((factor, idx) => (
                            <div key={idx} className="bg-red-50 border-l-4 border-red-500 p-3 rounded">
                              <p className="font-medium text-red-900 text-sm">{factor.text}</p>
                              <p className="text-xs text-red-700 mt-1">{factor.value}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {analysis.concerning.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-base font-semibold text-yellow-600 mb-2 flex items-center gap-2">
                          <AlertCircle className="w-5 h-5" />
                          Concerning Factors ({analysis.concerning.length})
                        </h4>
                        <div className="space-y-2">
                          {analysis.concerning.map((factor, idx) => (
                            <div key={idx} className="bg-yellow-50 border-l-4 border-yellow-500 p-3 rounded">
                              <p className="font-medium text-yellow-900 text-sm">{factor.text}</p>
                              <p className="text-xs text-yellow-700 mt-1">{factor.value}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {analysis.protective.length > 0 && (
                      <div>
                        <h4 className="text-base font-semibold text-green-600 mb-2 flex items-center gap-2">
                          <Shield className="w-5 h-5" />
                          Protective Factors ({analysis.protective.length})
                        </h4>
                        <div className="space-y-2">
                          {analysis.protective.map((factor, idx) => (
                            <div key={idx} className="bg-green-50 border-l-4 border-green-500 p-3 rounded">
                              <p className="font-medium text-green-900 text-sm">{factor.text}</p>
                              <p className="text-xs text-green-700 mt-1">{factor.value}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Clinical Recommendations */}
                <div className="bg-white rounded-2xl shadow-xl p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">Personalized Clinical Recommendations</h3>
                  
                  {(() => {
                    const recs = getClinicalRecommendations(result, input, analysis);
                    return (
                      <div className="space-y-4">
                        {recs.immediate.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-red-600 text-sm mb-2 uppercase tracking-wide">Immediate Actions</h4>
                            <ul className="space-y-1.5">
                              {recs.immediate.map((rec, idx) => (
                                <li key={idx} className="text-sm text-gray-700 pl-4 border-l-2 border-red-300 py-1">• {rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {recs.monitoring.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-blue-600 text-sm mb-2 uppercase tracking-wide">Monitoring & Follow-up</h4>
                            <ul className="space-y-1.5">
                              {recs.monitoring.map((rec, idx) => (
                                <li key={idx} className="text-sm text-gray-700 pl-4 border-l-2 border-blue-300 py-1">• {rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {recs.treatment.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-purple-600 text-sm mb-2 uppercase tracking-wide">Treatment Interventions</h4>
                            <ul className="space-y-1.5">
                              {recs.treatment.map((rec, idx) => (
                                <li key={idx} className="text-sm text-gray-700 pl-4 border-l-2 border-purple-300 py-1">• {rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {recs.lifestyle.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-green-600 text-sm mb-2 uppercase tracking-wide">Lifestyle Modifications</h4>
                            <ul className="space-y-1.5">
                              {recs.lifestyle.map((rec, idx) => (
                                <li key={idx} className="text-sm text-gray-700 pl-4 border-l-2 border-green-300 py-1">• {rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {recs.support.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-orange-600 text-sm mb-2 uppercase tracking-wide">Support Resources</h4>
                            <ul className="space-y-1.5">
                              {recs.support.map((rec, idx) => (
                                <li key={idx} className="text-sm text-gray-700 pl-4 border-l-2 border-orange-300 py-1">• {rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>

                {/* Disclaimer */}
                <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-4">
                  <p className="text-xs text-blue-900 text-center">
                    <strong>Medical Disclaimer:</strong> This assessment is a clinical decision support tool and should not replace professional medical judgment. Always consult qualified healthcare professionals for diagnosis and treatment decisions.
                  </p>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="text-center mt-8 text-gray-500 text-xs">
          <p>MediVision © 2024 • Advanced Machine Learning Risk Assessment System</p>
        </div>

      </div>
    </div>
  );
}