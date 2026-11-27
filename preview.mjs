import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const root=new URL('./ui/',import.meta.url);
const routes={'/':'index.html','/index.html':'index.html','/style.css':'style.css','/app.js':'app.js'};
const mime={html:'text/html; charset=utf-8',css:'text/css; charset=utf-8',js:'text/javascript; charset=utf-8'};
http.createServer(async(req,res)=>{const name=routes[new URL(req.url,'http://localhost').pathname];if(!name){res.writeHead(404);res.end('Not found');return;}try{const body=await readFile(fileURLToPath(new URL(name,root)));res.writeHead(200,{'Content-Type':mime[name.split('.').pop()],'Cache-Control':'no-store'});res.end(body);}catch{res.writeHead(500);res.end('Unable to load preview');}}).listen(4173,'127.0.0.1',()=>console.log('PolicyAI preview: http://localhost:4173'));
