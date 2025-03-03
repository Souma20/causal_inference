import React from 'react';
import { 
    Paper, 
    Typography, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow,
    Tooltip,
    Box
} from '@mui/material';

const ConfounderAnalysis = ({ data }) => {
    if (!data?.top_confounders) {
        return null;
    }

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Top Confounding Variables
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Variables with strongest relationships to both treatment assignment and outcome
            </Typography>
            
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Variable</TableCell>
                            <Tooltip title="Correlation with treatment assignment">
                                <TableCell align="right">Treatment Corr.</TableCell>
                            </Tooltip>
                            <Tooltip title="Correlation with outcome">
                                <TableCell align="right">Outcome Corr.</TableCell>
                            </Tooltip>
                            <Tooltip title="Statistical significance (p-value)">
                                <TableCell align="right">P-value</TableCell>
                            </Tooltip>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {data.top_confounders.map((confounder, index) => (
                            <TableRow 
                                key={confounder.covariate}
                                sx={{ 
                                    backgroundColor: index === 0 ? 'rgba(25, 118, 210, 0.08)' : 'inherit',
                                    '&:hover': {
                                        backgroundColor: 'rgba(25, 118, 210, 0.12)'
                                    }
                                }}
                            >
                                <TableCell component="th" scope="row">
                                    {confounder.covariate}
                                </TableCell>
                                <TableCell align="right">
                                    {confounder.treatment_corr.toFixed(3)}
                                </TableCell>
                                <TableCell align="right">
                                    {confounder.outcome_corr.toFixed(3)}
                                </TableCell>
                                <TableCell align="right">
                                    {confounder.p_value < 0.001 ? '< 0.001' : confounder.p_value.toFixed(3)}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
            
            <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                    * Highlighted row indicates the strongest confounder
                </Typography>
            </Box>
        </Paper>
    );
};

export default ConfounderAnalysis; 