import './App.css';

import * as React from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid2';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import {red, lightGreen, green} from '@mui/material/colors';
import Divider from '@mui/material/Divider';
import Slider from '@mui/material/Slider';
import CircularProgress from '@mui/material/CircularProgress';

import stages from './realStations.json';


const importantIndex = 85;

const findByIndex = (index) => {
    stages.forEach((stage) => {
        stage.sensors.forEach((sensor) => {
            if (sensor.index === index) {
                return sensor;
            }
        });
    });
};


const scale = (n) => {
    if (n == 0) return 0;
    else if (n < 0.025) return 50;
    else if (n < 0.05) return 100;
    else if (n < 0.1) return 200;
    else if (n < 0.15) return 300;
    else if (n < 0.2) return 400;
    else if (n < 0.3) return 500;
    else if (n < 0.4) return 600;
    else if (n < 0.6) return 700;
    else if (n < 0.75) return 800;
    else return 900;
}





function App() {
    const [focused, setFocused] = React.useState();
    const [switchValue, setSwitchValue] = React.useState(true);
    const [thresholdValue, setThresholdValue] = React.useState(0.5);
    const [loading, setLoading] = React.useState(true);

    let shownIndex = focused != null ? focused : importantIndex;

    const handleSwitch = () => {
        setSwitchValue(!switchValue);
    }

    const handleChange = (event, nextFocused) => {
        setFocused(nextFocused);
    };

    const valuetext = (value) => {
        setThresholdValue(value);
        return value;
    }

    React.useEffect(() => {
        const timer = setTimeout(() => {
            setLoading(false); // This will run after 5 seconds
        }, 3000);
        return () => clearTimeout(timer); // Clean up the timer
    }, []);



    if (loading) {
        return (
            <Box container justifyContent="space-around" alignContent="center"
                 sx={{display: 'flex', flexGrow: 3, height: "100%"}}>
                <Box>
                    <CircularProgress/>
                </Box>
            </Box>
        );
    }

    return (
        <Box sx={{flexGrow: 2, overflow: "visible"}}>
            <Grid container display="flex" spacing={3} justifyContent="space-between" alignContent="center"
                  sx={{width: "100%", height: "50px", bgcolor: 'primary.main', px: "10%"}}>
                <Grid container display="flex" alignContent="center" justifyContent="space-between"  sx={{height: "50px", width:"45%"}}>
                    <Grid alignContent="center" sx={{color: 'white', ml:"10px" , '& .MuiFormControlLabel-label': {color: 'white'}}}>
                        Sensor: {shownIndex}
                    </Grid>
                    <Grid>
                        <FormControlLabel
                            sx={{color: 'white', '& .MuiFormControlLabel-label': {color: 'white'}}}
                            label={switchValue ? "Caused by" : "Causes"} labelPlacement="end" labelColor="white"
                            control={
                                <Switch defaultChecked
                                        onChange={handleSwitch}
                                        sx={{
                                            '& .MuiSwitch-switchBase.Mui-checked': {
                                                color: red[50]
                                            },
                                            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                backgroundColor: red[50]
                                            }
                                        }}
                                />}
                        />
                    </Grid>
                </Grid>
                <Grid container display="flex" alignContent="center" justifyContent="space-between"  sx={{height: "50px", width: "45%"}}>
                    <Grid alignContent="center" sx={{color: 'white', '& .MuiFormControlLabel-label': {color: 'white'}}}>
                        Filter Cuttoff: {thresholdValue}
                    </Grid>
                    <Grid justifyContent="flex-end" sx={{width: 200, pt: "5px"}}>
                        <Slider
                            sx={{
                                color: thresholdValue < 0.05 ? red[50] : red[scale(thresholdValue)],

                            }}
                            getAriaValueText={valuetext}
                            defaultValue={0}
                            step={0.025}
                            marks
                            min={0}
                            max={0.9}
                        />
                    </Grid>
                </Grid>
            </Grid>
            <Grid container spacing={2}
                  justifyContent="center"
            >
                {stages.map((stage, stageIndex) =>
                    <Grid size={Math.floor(12 / stages.length)}>
                        <Accordion defaultExpanded >
                            <AccordionSummary
                                expandIcon={<ExpandMoreIcon/>}
                                aria-controls="panel1-content"
                                id="panel1-header"

                            >
                                {stage.name}
                            </AccordionSummary>
                            <AccordionDetails>
                                <ToggleButtonGroup
                                    orientation="vertical"
                                    value={shownIndex}
                                    color="primary"
                                    exclusive
                                    onChange={handleChange}
                                    sx={{width: '100%'}}
                                >

                                    {stage.sensors.map((sensor, sensorIndex) =>
                                        <ToggleButton value={sensor.index} sx={{
                                            backgroundColor: red[scale(switchValue ? sensor.causes[shownIndex] : sensor.causedBy[shownIndex])],
                                            display: shownIndex != sensor.index && (switchValue ? sensor.causes[shownIndex] : sensor.causedBy[shownIndex]) < thresholdValue ? "none" : "block"
                                        }}>
                                            <Grid container display="flex" justifyContent="space-between" width="100%">
                                                <Grid display="flex"
                                                      alignItems="center"> {sensor.name} </Grid>

                                                <Grid display="flex"
                                                      alignItems="center">{Math.round((switchValue ? sensor.causes[shownIndex] : sensor.causedBy[shownIndex]) * 1000) / 1000}</Grid>
                                            </Grid>
                                        </ToggleButton>
                                    )}

                                </ToggleButtonGroup>
                            </AccordionDetails>
                        </Accordion>
                    </Grid>
                )}
            </Grid>
        </Box>
    );

}

export default App;
