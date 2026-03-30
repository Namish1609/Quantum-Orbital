import re

with open('src/App.js', 'r', encoding='utf-8') as f:
    text = f.read()

if "import SplitPane" not in text:
    text = text.replace("import React, { useState, useEffect, useMemo } from 'react';", 
                        "import React, { useState, useEffect, useMemo } from 'react';\nimport SplitPane from 'react-split-pane';")

# Remove drag hooks
text = re.sub(r'  const \[graphHeight.*?useState\(250\);\n', '', text)
text = re.sub(r'  const \[isDragging.*?useState\(false\);\n', '', text)

# Remove useEffect for dragging
effect_regex = r'  useEffect\(\(\) => \{\n    if \(isDragging\).*?\}, \[isDragging\]\);\n'
text = re.sub(effect_regex, '', text, flags=re.DOTALL)

# Replace the layout
old_layout = r'''        <div className="main-content">
          <div className="canvas-container">
            <Canvas dpr={[1, 2]} gl={{ antialias: true, alpha: false }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                <color attach="background" args={['#111111']} />
                <ambientLight intensity={0.5} />
                <pointLight position={[100, 100, 100]} intensity={1} />
                <pointLight position={[-100, -100, -100]} intensity={0.5} />

                {plotData && (mode === 'scatter' ? (
                  <ScatterPlot key={renderKey} data={plotData} pointSize={pointSize} opacity={scatterOpacity} showPhase={showPhase} />
                ) : (
                  <IsosurfaceMesh key={renderKey} data={plotData} opacity={isoOpacity} showPhase={showPhase} isovalue={isovalue} />
                ))}

                <FullAxes size={gridSize} />
                <OrbitControls 
                  makeDefault 
                  enablePan={true} 
                  enableZoom={true} 
                  enableRotate={true}
                  autoRotate={true}
                  autoRotateSpeed={0.5}
                  panSpeed={1.5}
                />
            </Canvas>
          </div>

          <div style={{ height: '30px', backgroundColor: '#111', color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', borderTop: '1px solid #333' }}>
            <span><strong style={{color:'#ccc'}}>Controls:</strong> Left-Click to Rotate &bull; Right-Click to Pan/Move Target &bull; Scroll to Zoom</span>
          </div>

          <div className="radial-graph-container" style={{ height: `${graphHeight}px`, minHeight: '150px', backgroundColor: '#1a1a1a', borderTop: '2px solid #333', display: 'flex', flexDirection: 'column', position: 'relative' }}>
            {/* Drag Handle */}
            <div
              onMouseDown={() => setIsDragging(true)}
              style={{
                height: '8px',
                width: '100%',
                cursor: 'ns-resize',
                backgroundColor: isDragging ? '#00aaff' : '#333',
                position: 'absolute',
                top: '-4px',
                zIndex: 10,
                transition: 'background-color 0.2s'
              }}
              title="Drag to resize graph"
            />'''

new_layout = r'''        <div className="main-content">
          <SplitPane split="horizontal" defaultSize="65%" minSize={200} maxSize={-150} style={{ position: 'relative', height: '100%' }}>
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div className="canvas-container" style={{ flex: 1, backgroundColor: '#111', position: 'relative' }}>
                <Canvas dpr={[1, 2]} gl={{ antialias: true, alpha: false }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                    <color attach="background" args={['#111111']} />
                    <ambientLight intensity={0.5} />
                    <pointLight position={[100, 100, 100]} intensity={1} />
                    <pointLight position={[-100, -100, -100]} intensity={0.5} />

                    {plotData && (mode === 'scatter' ? (
                      <ScatterPlot key={renderKey} data={plotData} pointSize={pointSize} opacity={scatterOpacity} showPhase={showPhase} />
                    ) : (
                      <IsosurfaceMesh key={renderKey} data={plotData} opacity={isoOpacity} showPhase={showPhase} isovalue={isovalue} />
                    ))}

                    <FullAxes size={gridSize} />
                    <OrbitControls 
                      makeDefault 
                      enablePan={true} 
                      enableZoom={true} 
                      enableRotate={true}
                      autoRotate={true}
                      autoRotateSpeed={0.5}
                      panSpeed={1.5}
                    />
                </Canvas>
              </div>

              <div style={{ height: '30px', backgroundColor: '#111', color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', borderTop: '1px solid #333' }}>
                <span><strong style={{color:'#ccc'}}>Controls:</strong> Left-Click to Rotate &bull; Right-Click to Pan/Move Target &bull; Scroll to Zoom</span>
              </div>
            </div>

            <div className="radial-graph-container" style={{ height: '100%', backgroundColor: '#1a1a1a', borderTop: '2px solid #333', display: 'flex', flexDirection: 'column', position: 'relative' }}>'''

text = text.replace(old_layout, new_layout)

closing_old = r'''            </div>
          </div>
        </div>
      </div>
    </div>
  );'''

closing_new = r'''            </div>
          </SplitPane>
        </div>
      </div>
    </div>
  );'''
text = text.replace(closing_old, closing_new)

with open('src/App.js', 'w', encoding='utf-8') as f:
    f.write(text)
    
print("Replaced successfully")
