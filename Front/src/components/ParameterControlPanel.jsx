import React, { useState } from 'react';
import {
    Paper,
    Typography,
    Slider,
    TextField,
    Box,
    Tooltip,
    IconButton,
    Collapse,
    Alert,
    Grid,
    Chip,
    CircularProgress,
    Button
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import SaveIcon from '@mui/icons-material/Save';

const parameterInfo = {
    numNeighbors: {
        title: "Number of Neighbors (PSM)",
        description: "Number of control units to match with each treated unit.",
        min: 1,
        max: 10,
        step: 1,
        marks: [
            { value: 1, label: '1', description: 'Most precise, highest variance' },
            { value: 5, label: '5', description: 'Balanced precision/variance' },
            { value: 10, label: '10', description: 'Most stable, potential bias' }
        ],
        effects: {
            low: "🎯 High precision, but less stable estimates",
            medium: "⚖️ Balanced between precision and stability",
            high: "📊 More stable, but potential bias increase"
        }
    },
    trimWeights: {
        title: "Trim Weights (IPW)",
        description: "Threshold for trimming extreme propensity scores.",
        min: 0.01,
        max: 0.2,
        step: 0.01,
        marks: [
            { value: 0.01, label: '0.01', description: 'Minimal trimming' },
            { value: 0.1, label: '0.1', description: 'Moderate trimming' },
            { value: 0.2, label: '0.2', description: 'Heavy trimming' }
        ],
        effects: {
            low: "⚠️ More variance, potentially extreme weights",
            medium: "✅ Balanced between bias and variance",
            high: "🛡️ More stable, but potential information loss"
        }
    }
};

const ParameterControlPanel = ({ onParameterChange, initialParameters, isLoading }) => {
    const [parameters, setParameters] = useState(initialParameters);
    const [expanded, setExpanded] = useState(false);
    const [sensitivityMode, setSensitivityMode] = useState(false);
    const [tempParameters, setTempParameters] = useState(initialParameters);

    const getEffectLevel = (param, value) => {
        const info = parameterInfo[param];
        if (value <= (info.max - info.min) * 0.3 + info.min) return 'low';
        if (value <= (info.max - info.min) * 0.7 + info.min) return 'medium';
        return 'high';
    };

    const handleSliderChange = (param) => (event, newValue) => {
        setTempParameters(prev => ({
            ...prev,
            [param]: newValue
        }));
    };

    const handleTextChange = (param) => (event) => {
        const value = Number(event.target.value);
        const info = parameterInfo[param];
        
        if (!isNaN(value) && value >= info.min && value <= info.max) {
            setTempParameters(prev => ({
                ...prev,
                [param]: value
            }));
        }
    };

    const handleKeyDown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleApplyChanges();
        }
    };

    const handleApplyChanges = () => {
        setParameters(tempParameters);
        onParameterChange(tempParameters);
    };

    const handleSensitivityAnalysis = () => {
        setSensitivityMode(true);
        // Run analysis with different parameter values
        const sensitivityParams = [
            { numNeighbors: 1, trimWeights: parameters.trimWeights },
            { numNeighbors: 5, trimWeights: parameters.trimWeights },
            { numNeighbors: 10, trimWeights: parameters.trimWeights },
            { numNeighbors: parameters.numNeighbors, trimWeights: 0.05 },
            { numNeighbors: parameters.numNeighbors, trimWeights: 0.1 },
            { numNeighbors: parameters.numNeighbors, trimWeights: 0.15 }
        ];
        
        // Run each parameter combination
        sensitivityParams.forEach(params => {
            onParameterChange(params);
        });
    };

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="h6" sx={{ mr: 1 }}>
                        Parameter Control Panel
                    </Typography>
                    <Tooltip title="Adjust parameters to fine-tune the analysis">
                        <InfoIcon color="action" fontSize="small" />
                    </Tooltip>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {isLoading && (
                        <CircularProgress size={20} thickness={4} sx={{ opacity: 0.5 }} />
                    )}
                    <IconButton onClick={() => setExpanded(!expanded)}>
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>
            </Box>

            <Collapse in={expanded}>
                <Alert severity="info" sx={{ mb: 2 }}>
                    Adjust parameters and press Enter or click Apply to update results.
                </Alert>

                <Grid container spacing={3}>
                    {Object.entries(parameterInfo).map(([param, info]) => (
                        <Grid item xs={12} key={param}>
                            <Box sx={{ mb: 3 }}>
                                <Typography variant="subtitle1" gutterBottom>
                                    {info.title}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                    {info.description}
                                </Typography>
                                
                                <Box sx={{ mb: 2 }}>
                                    <Chip 
                                        label={info.effects[getEffectLevel(param, tempParameters[param])]}
                                        color={getEffectLevel(param, tempParameters[param]) === 'medium' ? 'success' : 'warning'}
                                        size="small"
                                    />
                                </Box>

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Slider
                                        value={tempParameters[param]}
                                        onChange={handleSliderChange(param)}
                                        min={info.min}
                                        max={info.max}
                                        step={info.step}
                                        marks={info.marks.map(mark => ({
                                            ...mark,
                                            label: (
                                                <Tooltip title={mark.description}>
                                                    <span>{mark.label}</span>
                                                </Tooltip>
                                            )
                                        }))}
                                        valueLabelDisplay="auto"
                                        sx={{ flexGrow: 1 }}
                                    />
                                    <TextField
                                        value={tempParameters[param]}
                                        onChange={handleTextChange(param)}
                                        onKeyDown={handleKeyDown}
                                        type="number"
                                        inputProps={{
                                            min: info.min,
                                            max: info.max,
                                            step: info.step
                                        }}
                                        sx={{ width: 100 }}
                                    />
                                </Box>
                            </Box>
                        </Grid>
                    ))}
                </Grid>

                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                    <Button
                        variant="contained"
                        startIcon={<SaveIcon />}
                        onClick={handleApplyChanges}
                        disabled={isLoading}
                    >
                        Apply Changes
                    </Button>
                </Box>

                {sensitivityMode && (
                    <Alert severity="warning" sx={{ mt: 2 }}>
                        Running sensitivity analysis... Results will update automatically.
                    </Alert>
                )}
            </Collapse>
        </Paper>
    );
};

export default ParameterControlPanel;