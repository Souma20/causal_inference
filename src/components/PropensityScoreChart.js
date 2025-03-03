import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const PropensityScoreChart = ({ data, method, parameters }) => {
    if (!data?.propensity_scores?.treated || !data?.propensity_scores?.control) {
        return (
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography>No propensity score data available</Typography>
            </Paper>
        );
    }

    // Create histogram data
    const bins = 20;
    const treated = data.propensity_scores.treated;
    const control = data.propensity_scores.control;
    
    const chartData = {
        labels: Array.from({ length: bins }, (_, i) => (i / bins).toFixed(2)),
        datasets: [
            {
                label: 'Treated',
                data: treated,
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true,
                tension: 0.4
            },
            {
                label: 'Control',
                data: control,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
                tension: 0.4
            }
        ]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Propensity Score Distribution'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Density'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Propensity Score'
                }
            }
        }
    };

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Propensity Score Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
                <Line 
                    data={chartData} 
                    options={options} 
                    key={`propensity-${method}-${JSON.stringify(parameters)}`}
                />
            </Box>
        </Paper>
    );
};

export default PropensityScoreChart; 