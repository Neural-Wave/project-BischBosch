const fs = require('fs');
const readline = require('readline');

async function readCsvToMatrix(filePath) {
    const fileStream = fs.createReadStream(filePath);

    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    let matrix = [];

    for await (const line of rl) {
        matrix.push(line.split(',').map(item => item.trim())); // Split each line by comma and trim spaces
    }

    return matrix;
}

function constructCausationJson(matrix) {
    let stages = [];
    const titlesStr = matrix.shift();

    titlesStr.forEach((measurement, index) => {
        let [station, sensor, sensorNumber] = measurement.split("_");
        if (!stages[station]) {
            stages[station] = {
                name: station,
                sensors: [{index: sensorNumber, name: sensor + sensorNumber, causes: Object.values(matrix[index])}]
            };
        } else {
            stages[station].sensors.push({
                index: sensorNumber,
                name: sensor + sensorNumber,
                causes: Object.values(matrix[index])
            });
        }

    });
    return Object.values(stages);
}

function transpose(matrix) {
    matrix.shift();

    const rows = matrix.length;
    const cols = matrix[0].length;

    const transposed = new Array(cols).fill(null).map(() => new Array(rows));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

function addCausedBy(matrix, stages) {
    let transposed = transpose(matrix);
    stages.forEach(stage => {
        stage.sensors.forEach(sensor => {
            sensor.causedBy = transposed[sensor.index];
        });
    });

    return stages;
}


readCsvToMatrix('real.csv').then(matrix => {

    let stages = addCausedBy(matrix, constructCausationJson(matrix));

    fs.writeFile("realStations.json", JSON.stringify(stages), function (err) {
            if (err) throw err;
            console.log('completed');
        }
    );

});


