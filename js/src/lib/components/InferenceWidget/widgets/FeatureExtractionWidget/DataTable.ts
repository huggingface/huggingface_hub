export class DataTable {
	max: number;
	min: number;
	std: number;

	constructor(public body: number[] | number[][]) {
		const all = this.body.flat();
		this.max  = Math.max(...all);
		this.min  = Math.min(...all);
		this.std  = this.max - this.min;
	}

	get isArrLevel0(): boolean {
		return isArrLevel0(this.body);
	}

	get oneDim(): number[] {
		return this.body as number[];
	}
	get twoDim(): number[][] {
		return this.body as number[][];
	}

	bg(value: number): string {
		if (value > this.min + this.std * 0.7) {
			return "bg-green-100 dark:bg-green-800";
		}
		if (value > this.min + this.std * 0.6) {
			return "bg-green-50 dark:bg-green-900";
		}
		if (value < this.min + this.std * 0.3) {
			return "bg-red-100 dark:bg-red-800";
		}
		if (value < this.min + this.std * 0.4) {
			return "bg-red-50 dark:bg-red-900";
		}
		return "";
	}
}

function isArrLevel0(x: number[] | number[][]): x is number[] {
	return typeof x[0] === "number";
}
