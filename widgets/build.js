import * as svelte from 'svelte/compiler';
import * as fs from 'fs/promises';
import path from 'path';
import preprocess from "svelte-preprocess";

console.info('[svelte] building views...');

let errors = false;
const srcDir = './src/lib';
const distDir = './package';

await compileFiles(srcDir, distDir);
await updatePackageJson(distDir);
// await replaceTypesImport(distDir);

console.info('[svelte] build complete');

/**************************************************************/

async function compileFile(fname, srcDir, distDir) {
	try {
		const extension = path.extname(fname);
		const originFile = path.join(srcDir, fname);

		if (extension === '.svelte') {
			const destFile = path.join(distDir, fname + '.js');
			await fs.mkdir(path.dirname(destFile), { recursive: true });

			const content = await fs.readFile(originFile, 'utf8');
			const processed = await svelte.preprocess(content, preprocess({
				defaults: { script: 'typescript' },
				typescript: { tsconfigFile: `./tsconfig.svelte.json` }
			}), {
				filename: fname,
			});

			const { js, warnings, ast } = svelte.compile(processed.code, {
				css: false,
				format: 'cjs',
				generate: 'ssr',
				dev: false,
				hydratable: true,
				filename: fname,
			});

			if (ast.instance) {
				/// here if the file contains a `<script></script>` part
				const body = ast.instance.content.body;
				const svelteImports = body.filter(
					/// only get entries that are svelte file imports
					(x) =>
						x.type === 'ImportDeclaration' && x.source.value.endsWith('.svelte')
				);

				for (const i of svelteImports) {
					const p = path.resolve(path.dirname(originFile), i.source.value);
					/// for each import, check the file actually exists
					/// TODO(?): add a cache to optimize perfs?
					/// (a `time node build-svelte.mjs` does not show any significant difference)
					try {
						await fs.access(p);
					} catch (error) {
						errors = true;
						console.error(`Error while processing file ${fname}`);
						console.error(error);
					}
				}
			}

			for (const warning of warnings) {
				let log = console.warn;
				if (warning.code === 'missing-declaration') {
					log = console.error;
					errors = true;
				} else if (warning.code === 'unused-export-let') {
					/// Too noisy
					continue;
				}

				log(`\n${warning.code} in ${warning.filename}`);
				log(warning.message);
				log(warning.frame);
			}

			const code = js?.code?.replace(/(require\(".*\.)(svelte)("\);$)/gm, '$1$2.js$3') ?? '';
			await fs.writeFile(destFile, code, { encoding: 'utf-8' });
		}
	} catch (error) {
		errors = true;
		console.error(`Error while processing file ${fname}`);
		console.error(error);
	}
}

async function compileFiles(srcDir, distDir) {
	const filenames = (
		await readdirREnt(srcDir, (dirent) => dirent.isFile())
	).map((x) => path.relative(srcDir, x));
	for (const fname of filenames) {
		await compileFile(fname, srcDir, distDir);
	}
}

async function replaceTypesImport(distDir) {
	const filenames = (
		await readdirREnt(distDir, (dirent) => dirent.isFile())
	).map((x) => path.relative(distDir, x));
	for (const fname of filenames) {
		const originFile = path.join(distDir, fname);
		let content = await fs.readFile(originFile, 'utf8');
		if (content.includes('interfaces/Types')) {
			const destFile = path.join(distDir, fname);
			const typesPath = path.relative(destFile, `${distDir}/interfaces/Types`);
			content = content.replace(/["']([.\/in]+interfaces\/Types)["']/g, `"${typesPath.slice(3)}"`);
			await fs.writeFile(destFile, content, { encoding: 'utf-8' });
		}
	}
}

async function readdirREnt(dirpath, matchDirEnt, maxDepth) {
	const dirEnts = await fs.readdir(dirpath, {
		withFileTypes: true,
	});
	const children = dirEnts.filter(x => matchDirEnt(x))
		.map(x => path.join(dirpath, x.name))
		;
	if (maxDepth === undefined || maxDepth > 1) {
		const mD = !!maxDepth
			? maxDepth - 1
			: undefined
			;
		const descendants = (await Promise.all(
			dirEnts
				.filter(x => x.isDirectory())
				.map(x => readdirREnt(path.join(dirpath, x.name), matchDirEnt, mD))
		)).flat();

		return [
			...children,
			...descendants,
		];
	} else {
		return children;
	}
}

async function updatePackageJson(distDir) {
	const originalFile = './package.json';
	const destFile = `${distDir}/package.json`;
	let content = await fs.readFile(originalFile, 'utf8');
	content = content.replace(/\/package/g, '');
	const json = JSON.parse(content);
	const deleteKeys = ['scripts', 'devDependencies', 'type', 'files'];
	for (const key of deleteKeys) {
		delete json[key];
	};
	const jsonString = JSON.stringify(json);
	await fs.writeFile(destFile, jsonString, { encoding: 'utf-8' });
}

