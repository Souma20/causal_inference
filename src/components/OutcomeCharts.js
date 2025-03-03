import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const OutcomeCharts = ({ data, method, parameters }) => {
    if (!data?.data?.outcomes) {
        return (
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography>No outcome data available</Typography>
            </Paper>
        );
    }

    const chartData = {
        labels: data.data.labels,
        datasets: [{
            label: 'Outcome Comparison',
            data: data.data.outcomes,
            backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(153, 102, 255, 0.2)'],
            borderColor: ['rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)'],
            borderWidth: 1,
        }],
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
                text: 'Outcome Comparison Between Groups'
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Outcome Value'
                }
            }
        }
    };

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Outcome Comparison
            </Typography>
            <Box sx={{ height: 300 }}>
                <Bar 
                    data={chartData} 
                    options={options} 
                    key={`outcome-${method}-${JSON.stringify(parameters)}`}
                />
            </Box>
        </Paper>
    );
};

export default OutcomeCharts; 