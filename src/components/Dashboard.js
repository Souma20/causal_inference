import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import { Grid, Box, Typography, Alert, LinearProgress, Paper } from '@mui/material';
import debounce from 'lodash/debounce';
import SummaryCards from './SummaryCards';
import MethodSelector from './MethodSelector';
import OutcomeCharts from './OutcomeCharts';
import PropensityScoreChart from './PropensityScoreChart';
import CovariateBalance from './CovariateBalance';
import ParameterControlPanel from './ParameterControlPanel';
import ConfounderAnalysis from './ConfounderAnalysis';
import CateAnalysis from './CateAnalysis';
import DataTable from './DataTable';

const Dashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedMethod, setSelectedMethod] = useState('Naive');
    const [parameters, setParameters] = useState({
        numNeighbors: 5,
        trimWeights: 0.1
    });
    const [updateCounter, setUpdateCounter] = useState(0); // Add counter for forcing updates

    const fetchData = useCallback(async (method, params) => {
        if (!method || !params) return;
        
        setLoading(true);
        setError(null);
        try {
            console.log('Fetching data with:', { method, params });
            const response = await axios.post('http://localhost:5000/run-analysis', {
                csv_path: 'ihdp_data.csv',
                method,
                numNeighbors: Number(params.numNeighbors),
                trimWeights: Number(params.trimWeights),
                include_diagnostics: true  // Request additional diagnostic information
            });
            
            if (!response.data) {
                throw new Error('No data received from server');
            }
            
            console.log('Received data:', {
                method: response.data.method,
                hasOutcomes: !!response.data.data?.outcomes,
                hasPropensityScores: !!response.data.propensity_scores,
                hasBalancePlot: !!response.data.balance_plot,
                plotLength: response.data.balance_plot?.length,
                hasDiagnostics: !!response.data.diagnostics
            });
            
            setData(response.data);
            setUpdateCounter(prev => prev + 1);
        } catch (err) {
            console.error('Error fetching data:', err);
            setError(err.message || 'An error occurred while fetching data');
        } finally {
            setLoading(false);
        }
    }, []);

    // Debounce the parameter changes to prevent too frequent API calls
    const debouncedFetchData = useMemo(
        () => debounce((method, params) => fetchData(method, params), 500),
        [fetchData]
    );

    const handleMethodChange = useCallback((method) => {
        if (method === selectedMethod) return;
        console.log('Method changed to:', method);
        setSelectedMethod(method);
        debouncedFetchData(method, parameters);
    }, [parameters, debouncedFetchData, selectedMethod]);

    const handleParameterChange = useCallback((newParameters) => {
        console.log('Parameters changed to:', newParameters);
        setParameters(newParameters);
        debouncedFetchData(selectedMethod, newParameters);
    }, [selectedMethod, debouncedFetchData]);

    useEffect(() => {
        console.log('Initial data fetch');
        fetchData(selectedMethod, parameters);
    }, []);

    // Effect to log data changes
    useEffect(() => {
        if (data) {
            console.log('Data updated:', {
                method: selectedMethod,
                parameters,
                ate: data.ate,
                hasBalancePlot: !!data.balance_plot,
                hasPropensityPlot: !!data.propensity_plot
            });
        }
    }, [data]);

    const renderContent = useMemo(() => {
        const hasValidData = data && data.balance_plot && !loading;
        return (
            <Grid container spacing={3}>
                {loading && (
                    <Grid item xs={12}>
                        <LinearProgress 
                            sx={{ 
                                width: '100%', 
                                position: 'absolute', 
                                top: 0, 
                                left: 0,
                                opacity: 0.7
                            }} 
                        />
                    </Grid>
                )}

                {error && (
                    <Grid item xs={12}>
                        <Alert severity="error">
                            Error: {error}
                        </Alert>
                    </Grid>
                )}
                
                {/* Controls Section - Always show */}
                <Grid item xs={12} md={4}>
                    <MethodSelector 
                        onMethodChange={handleMethodChange} 
                        selectedMethod={selectedMethod} 
                        isLoading={loading}
                    />
                    <ParameterControlPanel 
                        onParameterChange={handleParameterChange} 
                        initialParameters={parameters}
                        isLoading={loading}
                    />
                    {data && (
                        <>
                            <ConfounderAnalysis data={data} />
                            {data.diagnostics && (
                                <Paper sx={{ p: 2, mt: 2 }}>
                                    <Typography variant="h6" gutterBottom>
                                        Analysis Diagnostics
                                    </Typography>
                                    {selectedMethod === 'PSM' && (
                                        <>
                                            <Typography variant="subtitle2">
                                                Matching Statistics:
                                            </Typography>
                                            <Box sx={{ pl: 2 }}>
                                                <Typography variant="body2">
                                                    • Matched pairs: {data.diagnostics.matched_pairs}
                                                </Typography>
                                                <Typography variant="body2">
                                                    • Average distance: {data.diagnostics.avg_match_distance?.toFixed(3)}
                                                </Typography>
                                            </Box>
                                        </>
                                    )}
                                    {selectedMethod === 'IPW' && (
                                        <>
                                            <Typography variant="subtitle2">
                                                Weighting Statistics:
                                            </Typography>
                                            <Box sx={{ pl: 2 }}>
                                                <Typography variant="body2">
                                                    • Effective sample size: {data.diagnostics.effective_sample_size?.toFixed(1)}
                                                </Typography>
                                                <Typography variant="body2">
                                                    • Max weight: {data.diagnostics.max_weight?.toFixed(3)}
                                                </Typography>
                                            </Box>
                                        </>
                                    )}
                                    <Typography variant="subtitle2" sx={{ mt: 1 }}>
                                        Model Performance:
                                    </Typography>
                                    <Box sx={{ pl: 2 }}>
                                        <Typography variant="body2">
                                            • ATE estimate: {data.diagnostics.ate?.toFixed(3)}
                                        </Typography>
                                        {data.diagnostics.ci && (
                                            <Typography variant="body2">
                                                • 95% CI: [{data.diagnostics.ci[0]?.toFixed(3)}, {data.diagnostics.ci[1]?.toFixed(3)}]
                                            </Typography>
                                        )}
                                    </Box>
                                </Paper>
                            )}
                        </>
                    )}
                </Grid>

                {/* Results Section */}
                <Grid item xs={12} md={8}>
                    {!hasValidData && !loading ? (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            Loading data... Please wait.
                        </Alert>
                    ) : (
                        <Box sx={{ opacity: loading ? 0.7 : 1, transition: 'opacity 0.2s' }}>
                            {hasValidData && (
                                <>
                                    <SummaryCards data={data} key={`summary-${updateCounter}`} />
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <OutcomeCharts 
                                                data={data} 
                                                key={`outcome-${updateCounter}`}
                                                method={selectedMethod}
                                                parameters={parameters}
                                            />
                                        </Grid>
                                        <Grid item xs={12}>
                                            <PropensityScoreChart 
                                                data={data} 
                                                key={`propensity-${updateCounter}`}
                                                method={selectedMethod}
                                                parameters={parameters}
                                            />
                                        </Grid>
                                        <Grid item xs={12}>
                                            <CovariateBalance 
                                                data={data} 
                                                key={`balance-${updateCounter}`}
                                                method={selectedMethod}
                                                parameters={parameters}
                                            />
                                        </Grid>
                                    </Grid>
                                </>
                            )}
                        </Box>
                    )}
                </Grid>

                {/* Data Table Section */}
                {data && (
                    <Grid item xs={12} sx={{ opacity: loading ? 0.7 : 1, transition: 'opacity 0.2s' }}>
                        <DataTable 
                            data={data} 
                            key={`table-${updateCounter}`}
                            method={selectedMethod}
                            parameters={parameters}
                        />
                    </Grid>
                )}
            </Grid>
        );
    }, [data, loading, error, selectedMethod, parameters, handleMethodChange, handleParameterChange, updateCounter]);

    return (
        <Box sx={{ p: 3, position: 'relative' }}>
            {renderContent}
        </Box>
    );
};

export default Dashboard; 