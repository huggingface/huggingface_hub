module.exports = {
	root:    true,
	parser:  "@typescript-eslint/parser",
	plugins: [
		"@typescript-eslint",
		// "align-assignments",
		"github",
		"import",
		"svelte3",
	],
	parserOptions: {
		ecmaVersion:         2019,
		sourceType:          "module",
		// tsconfigRootDir:     __dirname,
		// project:             ['./tsconfig.json'],
		extraFileExtensions: [".svelte"],
	},
	env: {
		es6:     true,
		browser: true,
	},
	// ignorePatterns: [".eslintrc.js"],
	overrides: [
		{
			files:   ["*.{js,mjs,ts}"],
			extends: [
				"eslint:recommended",
				"plugin:@typescript-eslint/recommended",
			],
			rules: {
				/// https://github.com/typescript-eslint/typescript-eslint/tree/master/packages/eslint-plugin
				"@typescript-eslint/brace-style":  ["error"],
				"@typescript-eslint/comma-dangle": ["error", {
					arrays:    "always-multiline",
					objects:   "always-multiline",
					imports:   "always-multiline",
					exports:   "always-multiline",
					enums:     "always-multiline",
					functions: "never",
				}],
				"@typescript-eslint/comma-spacing":          ["error"],
				"@typescript-eslint/keyword-spacing":        ["error"],
				"@typescript-eslint/member-delimiter-style": [
					"error", { singleline: { requireLast: true } },
				],
				"@typescript-eslint/object-curly-spacing": ["error", "always"],
				"@typescript-eslint/quotes":               [
					"error",
					"double",
					{ avoidEscape: true, allowTemplateLiterals: true },
				],
				"@typescript-eslint/semi":                    ["error", "always"],
				"@typescript-eslint/space-infix-ops":         ["error", { int32Hint: false }],
				"@typescript-eslint/type-annotation-spacing": ["error"],
				"@typescript-eslint/no-unused-vars":          ["warn", { args: "none" }],
				
				/// https://github.com/lucasefe/eslint-plugin-align-assignments
				// "align-assignments/align-assignments": ["error"],
				
				/// https://github.com/benmosher/eslint-plugin-import
				// "import/first": ["error"],
				"import/order": ["error"],
				
				/// https://eslint.org/docs/rules/
				"array-bracket-spacing":    ["error", "never"],
				"arrow-parens":             ["error", "as-needed"],
				"arrow-spacing":            ["error"],
				"block-spacing":            ["error"],
				"eol-last":                 ["error", "always"],
				"indent":                   ["error", "tab", { SwitchCase: 1 }],
				"key-spacing":              ["error", { align: "value" }],
				"linebreak-style":          ["error", "unix"],
				// "max-len":               [
				// 	"error", { code: 100, ignoreUrls: true, ignoreComments: true },
				// ],
				"no-constant-condition":    ["error", { checkLoops: false }],
				"no-mixed-spaces-and-tabs": ["error", "smart-tabs"],
				"no-trailing-spaces":       ["error", { skipBlankLines: true, ignoreComments: true }],
				"object-curly-newline":     ["error"],
				"object-property-newline":  ["error", { allowAllPropertiesOnSameLine: true }],
				"operator-linebreak":       [
					"error",
					"before",
					{ overrides: { "=": "after" } },
				],
				"quote-props":         ["error", "consistent-as-needed"],
				"space-in-parens":     ["error", "never"],
				"space-before-blocks": ["error"],
			},
		},
		{
			files:     ["*.svelte"],
			processor: "svelte3/svelte3",
			extends:   [
				"eslint:recommended",
				"plugin:@typescript-eslint/recommended",
				"prettier", // Disable all eslint style-related rules for svelte files
			],
			rules:    {},
			settings: {
				"svelte3/typescript": true,
			},
		},
		{
			files:   ["*.{js,mjs,ts,svelte}"],
			extends: [
				"plugin:import/recommended",
				"plugin:import/typescript",
			],
			rules: {
				"@typescript-eslint/class-literal-property-style": ["error", "fields"],
				"@typescript-eslint/ban-types":                    ["warn"],
				"@typescript-eslint/consistent-type-assertions":   [
					"warn",
					{
						assertionStyle:              "as",
						objectLiteralTypeAssertions: "allow-as-parameter",
					},
				],
				"@typescript-eslint/consistent-type-definitions": [
					"error", "interface",
				],
				"@typescript-eslint/explicit-module-boundary-types": ["error"],
				"@typescript-eslint/method-signature-style":         ["error"],
				"@typescript-eslint/naming-convention":              [
					"error",
					{
						selector:           "default",
						format:             null,
						leadingUnderscore:  "allowSingleOrDouble",
						trailingUnderscore: "allowSingleOrDouble",
					},
			
					{
						selector:           "variable",
						format:             ["camelCase", "UPPER_CASE"],
						leadingUnderscore:  "allowSingleOrDouble",
						trailingUnderscore: "allowSingleOrDouble",
					},
					{
						selector:           "typeLike",
						format:             ["PascalCase"],
						leadingUnderscore:  "allowSingleOrDouble",
						trailingUnderscore: "allowSingleOrDouble",
					},
					{
						selector: "enumMember",
						format:   ["camelCase", "PascalCase", "UPPER_CASE"],
					},
				],
				"@typescript-eslint/no-empty-function":      "off",
				"@typescript-eslint/no-inferrable-types":    "off",
				"@typescript-eslint/no-namespace":           "off",
				"@typescript-eslint/prefer-for-of":          ["warn"],
				"@typescript-eslint/prefer-optional-chain":  ["warn"],
				"@typescript-eslint/prefer-ts-expect-error": ["warn"],
				"@typescript-eslint/unified-signatures":     ["warn"],
				
				/// https://github.com/github/eslint-plugin-github
				"github/async-currenttarget":  ["error"],
				"github/async-preventdefault": ["error"],
				"github/array-foreach":        ["error"],
				"github/prefer-observers":     ["error"],
				"github/no-dataset":           ["off"],
				
				"eqeqeq":                ["error"],
				"curly":                 ["error", "all"],
				"no-empty":              ["warn"],
				"no-extra-boolean-cast": "off",
				"no-undef":              "off",
				"no-unneeded-ternary":   ["error", { defaultAssignment: false }],
				"no-useless-escape":     ["warn"],
				"no-var":                ["error"],
				"prefer-const":          ["warn"],
			},
		},
	],
};
