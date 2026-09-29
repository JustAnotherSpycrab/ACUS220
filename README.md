# ACUS220 — Acoustic Monitoring Indexing System with BirdNET v3

Course project for **ACUS220: Computational Acoustics with Python**, Instituto de Acústica, Universidad Austral de Chile.

**Authors:**
- Enzo Latino
- Josué Clark 

**Course Instructors:**
- Professor: Dr. Víctor Poblete
- Teaching Assistant: Carlos Duarte

---

## Overview

This is an implementation of the BirdNET v3 neural network bird-recognizing algorithm, created by the BirdNET-Team. It's purpose is to ingest a whole database's worth of audio and generate the bird species files as a base to build a catalog, as quickly as possible

The algorithm utilizes the ONNX runtime and should run on a GPU for best results, but has a CPU fallback in case a graphic accelerator card is not available. (Actual GPU functionality is in the works).

---

## Repository Structure
