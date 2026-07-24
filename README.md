# Squirrel Engine

A fast, yet advanced browser-based game engine, featuring a 3D engine, physics engine, and multiplayer via server-side reconciliation.

The 3D engine features standard rasterization, as well as an option to use ray tracing instead. 
Unlike most other browser-based 3D engines, such as Three.js, made entirely in JavaScript, this 3D engine was written in Rust, then compiled into WASM, allowing for much faster speeds.

The physics engine is a modified version of Rapier.js, currently a WIP

The multiplayer engine is currently a WIP
## Installation

Note: the build command is currently only written in Windows Powershell. A Linux alternative will be coming soon.

1. Clone the repository

```bash
  git clone https://github.com/firephrog/Squirrel-Engine
```
2. Run the build command

```bash
  ./build.ps1
```

Simple Tech Demonstration
```bash
  node serve.mjs
```