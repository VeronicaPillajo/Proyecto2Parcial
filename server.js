const express = require('express');
const app = express();
const hbs = require('hbs');
const bodyParser = require('body-parser');
require('./hbs/helpers');
const createCsvWriter = require("csv-writer").createObjectCsvWriter;
const client = require("./conexion")
const Json2csvParser = require("json2csv").Parser;
const fs = require("fs");
let { PythonShell } = require('python-shell')


const port = process.env.PORT || 3000;

app.use(express.static(__dirname + '/public'));
client.connect()
    // Express HBS engine
hbs.registerPartials(__dirname + '/views/parciales');
app.set('view engine', 'hbs');

// Body Parser
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.get('/', function(req, res) {
    res.render('home');
});

//############################################################################################
//Proceso de inscripcion del estudiante
app.get('/registro', (req, res) => {

    res.render('registro')

});


var someVar = 0;
app.post('/addusuario', (req, res) => {

    client.query("SELECT idusuario FROM usuarioa where nombre = $1", [req.body.nombre], (err, te) => {
        if (err) {
            console.log(err);
        }
        let post = (te.rows.length > 0) ? te.rows[0] : null;
        let postInString = JSON.stringify(post);
        postInString = postInString.replace('{', '')
        postInString = postInString.replace('}', '')
        postInString = postInString.replace(':', '')
        postInString = postInString.replace('idusuario', '')
        postInString = postInString.replace('""', '')
        console.log(postInString);
        client.query('INSERT INTO inscripcion(idusuario,direccion,estadoacademico,periodo) VALUES($1,$2,$3,$4) ', [postInString, req.body.direccion, req.body.estadoacademico, req.body.periodo], (err, usuario) => {
            if (err) {
                console.log(err);
            }
            res.redirect('inscripcion_correcta');
        });

    });
});

//Inscripcion correcta
app.get('/inscripcion_correcta', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('inscripcion_correcta');
});



//############################################################################################
//DOCENTE y Insercion de Tema
app.get('/docente', (req, res) => {
    client.query("SELECT * FROM usuarioa where perfil = $1", ['d'], (err, te) => {
        if (err) {
            console.log(err);
        }
        res.render('docente', { idusuario: te.rows });
    });
});

app.post('/addtema', (req, res) => {
    let idusu = req.body.idusuario;
    idusu = idusu.substring(0, 3);
    idusu = idusu.replace(',', '')
    idusu = idusu.replace('.', '')
    idusu = idusu.replace('-', '')
    idusu = idusu.replace('"', '')
    console.log(idusu);
    client.query('INSERT INTO temas(idusuario,numest,nombretema,area,tipo_tema) VALUES($1,$2,$3,$4,$5) ', [idusu, req.body.numest, req.body.nombretema, req.body.area, req.body.tipo_tema], (err, result) => {
        if (err) {
            return console.error('error running query', err);
        }

        res.redirect('tema_confi')
    });

});


app.get('/tema_confi', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('tema_confi');
});


//############################################################################################
//Unidad de Titulacion Asignacion de Revisor
app.get('/utitulacion', (req, res) => {
    client.query("SELECT * from temas", (err, te) => {
        if (err) {
            console.log(err);
        }

        client.query("SELECT * from usuarioa where perfil = $1", ['r'], (err, usuario) => {
            if (err) {
                console.log(err);
            }
            res.render("utitulacion", { temas: te.rows, docentes: usuario.rows });
        });

    });

});

app.post('/addrev', (req, res) => {
    let idtema = req.body.idtema;
    idtema = idtema.substring(0, 3);
    idtema = idtema.replace('.', '')
    idtema = idtema.replace('-', '')
    console.log(idtema)

    let idrev = req.body.idrevisor;
    idrev = idrev.substring(0, 3);
    idrev = idrev.replace('.', '')
    idrev = idrev.replace('-', '')
    console.log(idrev)
    client.query('INSERT INTO temas_revisores(idtema,idusuario) VALUES($1,$2) ', [idtema, idrev], (err, result) => {
        if (err) {
            return console.error('error running query', err);
        }
        client.query('INSERT INTO estado_tema(idtema,tipoestado) VALUES($1,$2) ', [idtema, 'R'], (err, result) => {
            if (err) {
                return console.error('error running query', err);
            }
            res.redirect('revisor_confi')
        });
    });
});


app.get('/revisor_confi', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('revisor_confi');
});



//############################################################################################
//Docente Revisor
app.get('/revisor', (req, res) => {
    // callback
    client.query('SELECT * FROM usuarioa,temas, temas_revisores, estado_tema where temas.idtema = temas_revisores.idtema and usuarioa.idusuario = temas_revisores.idusuario and temas.idtema = estado_tema.idtema and tipoestado = $1', ['R'], (err, result) => {
        if (err) {
            console.log(err.stack)
        } else {

            console.log(result.rows[1])
            res.render('revisor', { nametema: result.rows });
        }
    });

});

app.post('/addob', (req, res) => {
    const date = new Date();
    let idtemar = req.body.idtemar;
    idtemar = idtemar.substring(0, 3);
    idtemar = idtemar.replace('.', '')
    idtemar = idtemar.replace('-', '')
    console.log(idtemar)
    client.query('INSERT INTO observaciones(idtemar,descripcion,fecha) VALUES($1,$2,$3) ', [idtemar, req.body.descripcion, date], (err, result) => {
        if (err) {
            return console.error('error running query', err);
        }
        res.redirect('tema_confi')
    });
});

//Confirmacion insercion de Observaciones
app.get('/ob_regis', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('ob_regis');
});




//############################################################################################
//Aprobacion de Tema

app.get('/coordinacion', (req, res) => {
    // // callback

    client.query("SELECT * FROM temas, temas_revisores,observaciones,estado_tema where temas.idtema = temas_revisores.idtema and temas_revisores.idtemar = observaciones.idtemar and temas.idtema = estado_tema.idtema and tipoestado = $1", ['R'], (err, ob) => {
        if (err) {
            console.log(err);
        }
        res.render("coordinacion", { obs: ob.rows });
    });

});



app.post('/addestado', (req, res) => {
    let idtema = req.body.idtema;
    idtema = idtema.substring(0, 3);
    idtema = idtema.replace('.', '')
    idtema = idtema.replace('-', '')
    console.log(idtema)
    client.query("UPDATE estado_tema SET tipoestado=$1 WHERE idtema=$2", ['A', idtema], (err, ob) => {
        if (err) {
            console.log(err);
        }
        //res.send('Esta es mi primera web app');
        res.redirect('creardata');
    });

});



//############################################################################################
//Creacion de CSV
app.get('/creardata', (req, res) => {
    client.query("SELECT idestado,nombretema, area, tipo_tema, tipoestado FROM temas,estado_tema WHERE temas.idtema = estado_tema.idtema AND tipoestado=$1 Order by idestado", ['A'], (err, result) => {

        if (err) {
            console.log(err.stack);
        }
        //res.send('Esta es mi primera web app');
        res.render('creardata', { data: result.rows });
    });
});

app.post('/crearcsv', (req, res) => {
    client.query("SELECT idestado,nombretema, area, tipoestado FROM temas,estado_tema WHERE temas.idtema = estado_tema.idtema AND tipoestado=$1 Order by idestado", ['A'], (err, res) => {

        if (err) {
            console.log(err.stack);
        } else {
            const jsonData = JSON.parse(JSON.stringify(res.rows));
            console.log("jsonData", jsonData);

            const json2csvParser = new Json2csvParser({ header: true, withBOM: true });
            const csv = json2csvParser.parse(jsonData);

            // let csv2 = csv.replace('""/gi', '')
            // console.log(csv2)

            fs.writeFile("temas.csv", csv, function(error) {
                if (error) throw error;
                console.log("Write to temas.csv successfully!");
            });
        }
    });
    res.redirect('csv');
});



//############################################################################################
//Informacion de Documentos necesarios antes de realizar el proceso
app.get('/informacion', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('informacion');
});


app.get('/csv', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('csv');
});

app.get('/resultados', (req, res) => {
    //res.send('Esta es mi primera web app');
    res.render('resultados');
});



// //CODIGO PYTHON (SCRIPT)
// PythonShell.run("Script_Proyecto.py", null, function (err) {
//     if (err) throw err;
//     console.log('finished');
//   });


//############################################################################################
//Salida del Puerto en el que se esta ejecutando la app
app.listen(port, () => {
    console.log(`Escuchando peticiones en el puerto ${port}`);
});