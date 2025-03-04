import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';

const CovariateBalance = ({ data, method, parameters }) => {
    const [imageError, setImageError] = useState(false);

    useEffect(() => {
        console.log('CovariateBalance received data:', {
            hasData: !!data,
            hasBalancePlot: !!data?.balance_plot,
            method,
            parameters,
            plotLength: data?.balance_plot?.length
        });
        setImageError(false); // Reset error state when data changes
    }, [data, method, parameters]);

    if (!data) {
        return (
            <Paper sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 100 }}>
                    <CircularProgress />
                </Box>
            </Paper>
        );
    }

    if (!data.balance_plot || imageError) {
        return (
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography color="error">
                    No covariate balance plot data available for the current method and parameters
                </Typography>
            </Paper>
        );
    }

    const imageUrl = `data:image/png;base64,${data.balance_plot}`;

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Covariate Balance Plot
            </Typography>
            <Typography variant="subtitle2" gutterBottom>
                Standardized mean differences between treated and control groups
            </Typography>
            <Box sx={{ 
                height: 500,
                width: '100%',
                display: 'flex', 
                justifyContent: 'center',
                alignItems: 'center',
                overflow: 'hidden',
                backgroundColor: '#ffffff',
                p: 2
            }}>
                <img 
                    src={imageUrl}
                    alt="Covariate Balance Plot"
                    style={{ 
                        maxWidth: '100%', 
                        maxHeight: '100%', 
                        width: 'auto',
                        height: 'auto',
                        objectFit: 'contain',
                        display: 'block',
                        margin: 'auto'
                    }}
                    key={`balance-${method}-${JSON.stringify(parameters)}`}
                    onError={(e) => {
                        console.error('Error loading balance plot:', e);
                        setImageError(true);
                    }}
                />
            </Box>
        </Paper>
    );
};

export default CovariateBalance; 