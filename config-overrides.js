module.exports = function override(config, env) {
    config.module.rules = [
        ...config.module.rules,
        {
            resourceQuery: /raw/,
            type: 'asset/source'
        },
        {
            test: /\.csv$/,
            loader: 'csv-loader',
            options: {
                dynamicTyping: true,
                header: false,
                skipEmptyLines: true

            }
        }
    ]
    return config;
}