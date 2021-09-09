import fs from "fs";
import path from "path";

const copyRecursiveSync = (src, dest) => {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    if (isDirectory) {
      fs.mkdirSync(dest);
      fs.readdirSync(src).forEach(function(childItemName) {
        copyRecursiveSync(path.join(src, childItemName),
                          path.join(dest, childItemName));
      });
    } else {
      fs.copyFileSync(src, dest);
    }
  };

const pathtToJson = './package/package.json';
let pkg = fs.readFileSync(pathtToJson);
pkg = JSON.parse(pkg);
delete pkg.type;
fs.writeFileSync(pathtToJson, JSON.stringify(pkg));

copyRecursiveSync("../docs", "./package/docs");