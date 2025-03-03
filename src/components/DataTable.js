import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from '@mui/material';

const DataTable = ({ data }) => {
    if (!data || !data.columns || !data.rows) {
        return <Typography variant="body1">No data available</Typography>;
    }

    return (
        <TableContainer component={Paper}>
            <Typography variant="h6" component="div" gutterBottom>
                Data Table
            </Typography>
            <Table>
                <TableHead>
                    <TableRow>
                        {data.columns.map((column) => (
                            <TableCell key={column}>{column}</TableCell>
                        ))}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.rows.map((row, index) => (
                        <TableRow key={index}>
                            {row.map((cell, cellIndex) => (
                                <TableCell key={cellIndex}>{cell}</TableCell>
                            ))}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default DataTable; 