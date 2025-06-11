import { Vector2 } from "./vector2";

export class Mouse {
  protected m_canvas_container: HTMLDivElement | undefined;
  protected m_pos: Vector2;

  constructor(canvas_container?: HTMLDivElement) {
    this.m_pos = new Vector2();
    this.m_canvas_container = canvas_container;

    document.onmousemove = (event: MouseEvent) => {
      //console.log(this._timer.delta);
      this.handleMouseMove(event);
    };
  }

  protected handleMouseMove(event: MouseEvent) {
    let [x_scale, y_scale] = [1, 1];
    let [x0, y0] = [0, 0];

    if (this.m_canvas_container !== undefined) {
      const rect = this.m_canvas_container.getBoundingClientRect();
      [x_scale, y_scale] = [2 / rect.width, 2 / rect.height];
      [x0, y0] = [rect.left + rect.width / 2, rect.top + rect.height / 2];
    }

    this.m_pos.x = x_scale * (event.x - x0);
    this.m_pos.y = -y_scale * (event.y - y0);
  }

  public set canvas_container(canvas: HTMLDivElement) {
    this.m_canvas_container = canvas;
  }

  public get x(): number {
    return this.m_pos.x;
  }

  public get y(): number {
    return this.m_pos.y;
  }

  public get pos(): Vector2 {
    return this.m_pos;
  }
}
