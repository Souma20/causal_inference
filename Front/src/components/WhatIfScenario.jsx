import React, { useState, useCallback } from 'react';
import {
    Paper,
    Typography,
    TextField,
    Button,
    Grid,
    Box,
    Alert,
    Slider,
    Tooltip,
    CircularProgress
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, Legend } from 'recharts';

const WhatIfScenario = ({ onSimulate, isLoading, simulationResults }) => {
    const [covariates, setCovariates] = useState({
        x1: '',
        x2: '',
        x3: ''
        // Add more covariates as needed
    });

    const handleChange = (covariate) => (event) => {
        setCovariates(prev => ({
            ...prev,
            [covariate]: event.target.value
        }));
    };

    const handleSimulate = () => {
        // Filter out empty values
        const validCovariates = Object.entries(covariates)
            .filter(([_, value]) => value !== '')
            .reduce((acc, [key, value]) => ({
                ...acc,
                [key]: Number(value)
            }), {});
            
        onSimulate(validCovariates);
    };

    return (
        <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
                What-If Scenario Analysis
            </Typography>
            
            <Grid container spacing={2}>
                {/* Covariate inputs */}
                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                        Modify Covariates
                    </Typography>
                    {Object.keys(covariates).map(covariate => (
                        <TextField
                            key={covariate}
                            label={`Covariate ${covariate}`}
                            value={covariates[covariate]}
                            onChange={handleChange(covariate)}
                            type="number"
                            fullWidth
                            margin="dense"
                            size="small"
                        />
                    ))}
                    <Button
                        variant="contained"
                        onClick={handleSimulate}
                        disabled={isLoading}
                        sx={{ mt: 2 }}
                        fullWidth
                    >
                        {isLoading ? <CircularProgress size={24} /> : 'Simulate'}
                    </Button>
                </Grid>

                {/* Results visualization */}
                <Grid item xs={12} md={6}>
                    {simulationResults && (
                        <Box>
                            <Typography variant="subtitle2" gutterBottom>
                                Simulation Results
                            </Typography>
                            <Typography variant="body2">
                                Simulated ATE: {simulationResults.simulated_ate.toFixed(3)}
                            </Typography>
                            
                            {/* Add visualization of potential outcomes */}
                            <Box sx={{ mt: 2 }}>
                                <LineChart width={400} height={200} data={[
                                    {
                                        name: 'Control',
                                        value: simulationResults.potential_control[0]
                                    },
                                    {
                                        name: 'Treated',
                                        value: simulationResults.potential_treated[0]
                                    }
                                ]}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <ChartTooltip />
                                    <Legend />
                                    <Line type="monotone" dataKey="value" stroke="#8884d8" />
                                </LineChart>
                            </Box>
                        </Box>
                    )}
                </Grid>
            </Grid>
        </Paper>
    );
};

export default WhatIfScenario; 