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
    Box
} from '@mui/material';

const CateAnalysis = ({ data }) => {
    if (!data || !data.cate_analysis) {
        return null;
    }

    const { average_cate, cate_std, subgroups } = data.cate_analysis;

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Heterogeneous Treatment Effects
            </Typography>
            
            <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                    Overall CATE Statistics
                </Typography>
                <Typography>
                    Average CATE: {average_cate.toFixed(3)}
                </Typography>
                <Typography>
                    Standard Deviation: {cate_std.toFixed(3)}
                </Typography>
            </Box>

            <Typography variant="subtitle1" gutterBottom>
                Top Subgroup Effects
            </Typography>
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Covariate</TableCell>
                            <TableCell align="right">Split Value</TableCell>
                            <TableCell align="right">Effect (High)</TableCell>
                            <TableCell align="right">Effect (Low)</TableCell>
                            <TableCell align="right">Difference</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {subgroups.map((group) => (
                            <TableRow key={group.covariate}>
                                <TableCell component="th" scope="row">
                                    {group.covariate}
                                </TableCell>
                                <TableCell align="right">
                                    {group.split_value.toFixed(2)}
                                </TableCell>
                                <TableCell align="right">
                                    {group.cate_high.toFixed(3)}
                                </TableCell>
                                <TableCell align="right">
                                    {group.cate_low.toFixed(3)}
                                </TableCell>
                                <TableCell align="right">
                                    {group.difference.toFixed(3)}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
};

export default CateAnalysis; 