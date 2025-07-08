---
title: Freecad Automation
date: 2025-07-08
---

# Freecad Automation

---
author: Knowledge Base Automation System
created_at: '2025-07-04'
description: Documentation on Freecad Automation for cad_manufacturing/freecad_automation.md
title: Freecad Automation
updated_at: '2025-07-04'
version: 1.0.0
---

# FreeCAD Automation Guide

This guide covers how to automate 3D modeling tasks in FreeCAD using Python scripting. FreeCAD's Python API allows for programmatic creation and modification of 3D models, making it ideal for parametric and generative design.

## Table of Contents
- [1. Basic Concepts](#1-basic-concepts)
- [2. Creating Primitives](#2-creating-primitives)
- [3. Boolean Operations](#3-boolean-operations)
- [4. Parametric Design](#4-parametric-design)
- [5. Exporting Models](#5-exporting-models)
- [6. Advanced Techniques](#6-advanced-techniques)
- [7. Example Projects](#7-example-projects)

## 1. Basic Concepts

### The FreeCAD Document Structure
```python
import FreeCAD

# Create a new document
doc = FreeCAD.newDocument("MyDesign")

# Access the active document
doc = FreeCAD.ActiveDocument

# Save the document
doc.saveAs("/path/to/design.FCStd")
``````python
import Part

# Create a box: makeBox(length, width, height, [base], [direction])
box = Part.makeBox(10, 10, 10)  # 10x10x10mm cube
box_object = doc.addObject("Part::Feature", "MyCube")
box_object.Shape = box
``````python
# Create a cylinder: makeCylinder(radius, height, [pnt], [dir], [angle])
cylinder = Part.makeCylinder(5, 10)  # 5mm radius, 10mm height
cyl_object = doc.addObject("Part::Feature", "MyCylinder")
cyl_object.Shape = cylinder
``````python
# Create a sphere: makeSphere(radius, [angle1], [angle2], [angle3], [center])
sphere = Part.makeSphere(5)  # 5mm radius
sph_object = doc.addObject("Part::Feature", "MySphere")
sph_object.Shape = sphere
``````python
# Create two boxes
box1 = Part.makeBox(10, 10, 10)
box2 = Part.makeBox(10, 10, 10, FreeCAD.Vector(5, 0, 0))

# Union
fused = box1.fuse(box2)
fused_object = doc.addObject("Part::Feature", "FusedObject")
fused_object.Shape = fused
``````python
# Difference (cut)
cut = box1.cut(box2)
cut_object = doc.addObject("Part::Feature", "CutObject")
cut_object.Shape = cut
``````python
# Intersection
common = box1.common(box2)
common_object = doc.addObject("Part::Feature", "CommonObject")
common_object.Shape = common
``````python
import FreeCAD
import Part

def create_parametric_cylinder(radius, height, position=FreeCAD.Vector(0,0,0)):
    """Create a parametric cylinder with given dimensions and position."""
    cylinder = Part.makeCylinder(radius, height, position)
    return cylinder

# Example usage
radius = 5.0  # mm
height = 15.0  # mm
position = FreeCAD.Vector(10, 10, 0)

cylinder = create_parametric_cylinder(radius, height, position)
cyl_object = doc.addObject("Part::Feature", "ParametricCylinder")
cyl_object.Shape = cylinder
``````python
def update_cylinder(obj, radius, height, position):
    """Update cylinder dimensions."""
    cylinder = Part.makeCylinder(radius, height, position)
    obj.Shape = cylinder

# Update the cylinder
update_cylinder(cyl_object, 8.0, 20.0, FreeCAD.Vector(0, 0, 0))
``````python
import Mesh

# Export a single object
Mesh.export([cyl_object], "/path / to / export.stl")

# Export all objects in document
objects = doc.Objects
shapes = [obj.Shape for obj in objects]
Mesh.export(shapes, "/path / to / all_objects.stl"):
``````python
import Import

# Export a single object
cyl_object.Shape.exportStep("/path/to/export.step")

# Export all visible objects
for obj in doc.Objects:
    if hasattr(obj, 'Shape') and obj.Visibility:
        obj.Shape.exportStep(f"/path/to/{obj.Name}.step")
``````python
class ParametricGear:
    def __init__(self, module=1, teeth=20, width=5):
        self.module = module
        self.teeth = teeth
        self.width = width
        
    def create(self):
        """Create a parametric gear."""
        # Gear creation logic here
        gear = Part.makeGear(self.module, self.teeth, self.width)
        return gear

# Usage
gear_maker = ParametricGear(module=1, teeth=24, width=5)
gear = gear_maker.create()
gear_object = doc.addObject("Part::Feature", "MyGear")
gear_object.Shape = gear
``````python
def batch_create_gears():
    """Create multiple gears with different parameters."""
    gears = []
    for i in range(5):
        gear = ParametricGear(module=1, teeth=20+i*2, width=5)
        gear_obj = gear.create()
        gear_doc = doc.addObject("Part::Feature", f"Gear_{i}")
        gear_doc.Shape = gear_obj
        gear_doc.Placement.Base = FreeCAD.Vector(i*30, 0, 0)
        gears.append(gear_doc)
    return gears
``````python
def create_bracket(length, width, height, thickness, hole_diameter):
    """Create a parametric L-bracket with mounting holes."""
    # Base plate
    base = Part.makeBox(length, width, thickness)
    
    # Vertical plate
    vertical = Part.makeBox(thickness, width, height)
    vertical.translate(FreeCAD.Vector(0, 0, thickness))
    
    # Create holes
    hole = Part.makeCylinder(hole_diameter/2, thickness*3, 
                           FreeCAD.Vector(length/4, width/2, -thickness))
    hole2 = hole.copy()
    hole2.translate(FreeCAD.Vector(length/2, 0, 0))
    
    # Combine everything
    bracket = base.fuse(vertical)
    bracket = bracket.cut(hole).cut(hole2)
    
    return bracket

# Create and add to document
bracket = create_bracket(50, 30, 40, 5, 4)
bracket_object = doc.addObject("Part::Feature", "LBracket")
bracket_object.Shape = bracket
```