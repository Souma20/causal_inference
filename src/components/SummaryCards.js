import React from 'react';
import { Card, CardContent, Typography, Grid } from '@mui/material';

const SummaryCards = ({ data }) => {
    const metrics = [
        { title: 'Estimated ATE (Naive)', value: data?.naive_ate?.toFixed(3) || 'N/A' },
        { title: 'Estimated ATE (PSM)', value: data?.psm_ate?.toFixed(3) || 'N/A' },
        { title: 'Estimated ATE (IPW)', value: data?.ipw_ate?.toFixed(3) || 'N/A' }
    ];

    return (
        <Grid container spacing={2} sx={{ mb: 4 }}>
            {metrics.map((metric, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" component="div" gutterBottom>
                                {metric.title}
                            </Typography>
                            <Typography variant="h4" component="div" color="primary">
                                {metric.value}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            ))}
        </Grid>
    );
};

export default SummaryCards; 