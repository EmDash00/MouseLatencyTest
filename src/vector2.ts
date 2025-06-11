import * as nj from "numjs";

/**
 * Represents a 2D vector using a NdArray. The first element is the x component
 * and the second element is the y component.
 */
export class Vector2 {
  protected _data: nj.NdArray;

  constructor(x: number = 0, y: number = 0) {
    this._data = nj.array([x, y]);
  }

  public toString(): string {
    return this._data.toString();
  }

  /**
   * Gets the underlying NdArray data.
   */
  get data(): nj.NjArray<number> {
    return this._data;
  }

  get x(): number {
    return this._data.get(0); // Access the first element
  }

  set x(value: number) {
    this._data.set(0, value); // Set the first element
  }

  get y(): number {
    return this._data.get(1); // Access the second element
  }

  set y(value: number) {
    this._data.set(1, value); // Set the second element
  }
}
