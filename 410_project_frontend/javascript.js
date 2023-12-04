//html doesn't work with this file yet since html gives error: require (from line 4) is not defined
//I tried to switch to import but still got an error
//import { spawn as spawner } from 'child_process';
const spawner = require('child_process').spawn


//This code can be used to test if the connection between javascript.js and python.py works
const data_to_pass_in = 'Can you give me an overview on Probabilistic Latent Semantic Analysis'

console.log('Data sent to python script:', data_to_pass_in)

const python_process = spawner('python', ['./python.py', data_to_pass_in])

python_process.stdout.on('data', (data) => {
    console.log('Data recieved from python script: ', data.toString())
});

//The function will be used grab the users question and get its response
function buttonClicked(){
    var textarea = document.getElementById('question')
    var question = textarea.value
    var response = document.getElementById('response')
    response.innerHTML = "loading..."
    console.log("Clicked")

    //console.log('Data sent to python script:', question)

    const python_process = spawner('python', ['./python.py', question])

    python_process.stdout.on('data', (data) => {
        console.log('Data recieved from python script: ', data.toString())
        response.innerHTML = data
    });
}
