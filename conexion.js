const { Pool, Client } = require('pg');

//Coneccion Base de Datos
const client = new Client({
    user: 'postgres',
    host: 'localhost',
    database: 'md1_proyecto',
    password: 'verito',
    port: 5432,
})

module.exports = client;