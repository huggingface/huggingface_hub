import fs from "fs";

const path = './package/package.json';
let pkg = fs.readFileSync(path);
pkg = JSON.parse(pkg);
delete pkg.type;
fs.writeFileSync(path, JSON.stringify(pkg));