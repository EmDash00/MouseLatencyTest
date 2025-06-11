import { FILL_INTSTYLE, loadGRRuntime, MARKERTYPE } from "gr";
import { Mouse } from "./mouse";

import ModuleConfig from "gr";

import "./styles/style.css";

ModuleConfig.locateFile = function (filename: string) {
  if (filename.endsWith(".wasm")) {
    return "/static/libgr.wasm";
  }
};

const grRuntime = await loadGRRuntime();
let gr = new grRuntime.GRCanvas("canvas");

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const canvas_container = document.getElementById(
  "container-canvas",
) as HTMLDivElement;

function resizeCanvas() {
  // Set the width and height attributes based on the CSS size
  canvas.style.width = `${canvas_container.clientWidth}px`;
  canvas.style.height = `${canvas_container.clientHeight}px`;
  gr.select_canvas();
}

// Dynamically resize the canvas to fit the browsser window
const observer = new ResizeObserver(resizeCanvas);
observer.observe(canvas_container);

const mouse = new Mouse(canvas_container);

function draw() {
  gr.setwindow(-1, 1, -1, 1);
  gr.setviewport(0, 1, 0, 1);
  gr.clearws();

  gr.setfillintstyle(FILL_INTSTYLE.SOLID);
  gr.setfillcolorind(gr.inqcolorfromrgb(0, 0, 0));
  gr.fillrect(...gr.inqwindow());

  gr.setmarkercolorind(gr.inqcolorfromrgb(1, 0, 0));
  gr.setmarkertype(MARKERTYPE.SOLID_CIRCLE);
  gr.polymarker([mouse.x], [mouse.y]);
  gr.updatews();

  window.requestAnimationFrame(draw);
}

window.requestAnimationFrame(draw);
