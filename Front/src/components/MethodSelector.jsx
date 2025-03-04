import React from 'react';
import { 
    FormControl, 
    InputLabel, 
    Select, 
    MenuItem, 
    Paper, 
    Typography,
    Tooltip,
    Box
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const methodDescriptions = {
    Naive: `Simple comparison of means between treated and control groups. 
            Does not account for confounding variables.`,
    PSM: `Propensity Score Matching (PSM) matches treated units with similar control units 
          based on their probability of receiving treatment.`,
    IPW: `Inverse Probability Weighting (IPW) uses weights based on propensity scores 
          to create a balanced pseudo-population.`
};

const MethodSelector = ({ onMethodChange, selectedMethod }) => {
    const handleMethodChange = (event) => {
        event.preventDefault();
        onMethodChange(event.target.value);
    };

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ mr: 1 }}>
                    Method Selector
                </Typography>
                <Tooltip title="Choose a causal inference method to estimate treatment effects">
                    <InfoIcon color="action" fontSize="small" />
                </Tooltip>
            </Box>
            
            <FormControl fullWidth>
                <InputLabel id="method-select-label">Method</InputLabel>
                <Select
                    labelId="method-select-label"
                    id="method-select"
                    value={selectedMethod}
                    label="Method"
                    onChange={handleMethodChange}
                >
                    {Object.keys(methodDescriptions).map((method) => (
                        <MenuItem key={method} value={method}>
                            <Tooltip 
                                title={methodDescriptions[method]}
                                placement="right"
                            >
                                <span>{method}</span>
                            </Tooltip>
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            
            <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                    {methodDescriptions[selectedMethod]}
                </Typography>
            </Box>
        </Paper>
    );
};

export default MethodSelector;