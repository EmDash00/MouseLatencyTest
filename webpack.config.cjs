// @ts-ignore
const path = require("path");
const CopyWebpackPlugin = require("copy-webpack-plugin");

const { WebpackManifestPlugin } = require('webpack-manifest-plugin');
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: "./src/index.ts",
  mode: "development",
  target: "web",
  output: {
    filename: "main.js",
    path: path.resolve(__dirname, "dist"),
    assetModuleFilename: 'static/[name][ext]', // Output assets to a `static` folder with their original names
  },
  node: {
    __dirname: true, // Enable __dirname
    global: true
  },
  plugins: [
    // { publicPath: "" } is a hack for this issue: https://github.com/shellscape/webpack-manifest-plugin/issues/229
    new WebpackManifestPlugin({ publicPath: "" }),
    new HtmlWebpackPlugin({
        template: 'src/index.html', // Path to your HTML file
        filename: 'index.html', // Output file name
    })
  ],
  externals: {
      /*
       * Hack needed to get GR to compile since webpack compiles inside Node...
       * GR thinks it's running inside node.js instead of client-side in the browser.
       * and then will try to require("module")
       * To do ESModule loading in node.js
       */
      module: 'module',
  },
  experiments: {
    futureDefaults: true,
    asyncWebAssembly: true,
    syncWebAssembly: true
  },
  module: {
    rules: [
      {
        test: /\.m?js$/,
        use: {
          loader: "babel-loader",
          options: {
            presets: ["@babel/preset-env"],
          },
        },
        resolve: {
          fullySpecified: false, // This prevents Webpack from requiring file extensions
        },
      },
      {
        test: /\.ts?$/,
        use: "ts-loader",
        exclude: "/node_modules",
      }
    ],
  },
  resolve: {
    modules: [path.join(__dirname, "node_modules")],
    extensions: [".tsx", ".ts", ".js", ".mjs", ".wasm", ".json"],
    fallback: {
      path: require.resolve("path-browserify"),
      process: require.resolve('process/browser'),
    }
  },
  devServer: {
    static: [
      path.join(__dirname, "dist"),
    ],
    compress: true,
    port: 4000,
  },
};
