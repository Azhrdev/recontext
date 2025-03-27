#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive 3D viewer for semantically-enriched scenes.

This module provides an interactive visualization interface for exploring
3D reconstructions with semantic labels and querying scene understanding.

Author: Alex Johnson
Date: 2024-02-15
Last modified: 2024-03-10
"""

import os
import sys
import numpy as np
import open3d as o3d
import logging
import json
import threading
import time
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Callable

# GUI imports (support either PyQt5 or PySide2)
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QComboBox, 
                               QLineEdit, QCheckBox, QTabWidget, QTextEdit,
                               QSplitter, QGroupBox, QListWidget, QSlider,
                               QFileDialog, QMessageBox, QAction, QMenu, QToolBar)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QIcon
    using_pyqt = True
except ImportError:
    try:
        from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                     QHBoxLayout, QLabel, QPushButton, QComboBox, 
                                     QLineEdit, QCheckBox, QTabWidget, QTextEdit,
                                     QSplitter, QGroupBox, QListWidget, QSlider,
                                     QFileDialog, QMessageBox, QAction, QMenu, QToolBar)
        from PySide2.QtCore import Qt, QThread, Signal as pyqtSignal
        from PySide2.QtGui import QColor, QIcon
        using_pyqt = False
    except ImportError:
        logging.error("Either PyQt5 or PySide2 is required for the interactive viewer")
        sys.exit(1)

# Import RECONTEXT modules
from recontext.language.scene_graph import SceneGraph, Object3D, Relationship
from recontext.language.query_engine import QueryEngine, NLQueryResult
from recontext.utils.io_utils import ensure_dir
from recontext.utils.transforms import normalize_pointcloud

logger = logging.getLogger(__name__)

# Default visualization settings
DEFAULT_SETTINGS = {
    'point_size': 2.0,
    'background_color': [0.1, 0.1, 0.1],
    'show_labels': True,
    'label_size': 16,
    'show_bounding_boxes': True,
    'show_axes': True,
    'highlight_selected': True,
    'highlight_color': [1.0, 0.5, 0.0],
}


class AsyncTask(QThread):
    """Thread for running async tasks without blocking the GUI."""
    
    finished = pyqtSignal(object)
    progress = pyqtSignal(float, str)
    error = pyqtSignal(str)
    
    def __init__(self, task_fn, *args, **kwargs):
        """Initialize async task.
        
        Args:
            task_fn: Function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Run the task."""
        try:
            result = self.task_fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Error in async task: {str(e)}")
            self.error.emit(str(e))


class ViewerState:
    """State container for the interactive viewer."""
    
    def __init__(self):
        """Initialize viewer state."""
        # Data
        self.pointcloud = None  # Original point cloud
        self.mesh = None  # Original mesh
        self.scene_graph = None  # Scene graph
        
        # Visualization objects
        self.vis_objects = {}  # ID -> Visualization objects
        self.selected_objects = set()  # Set of selected object IDs
        self.highlighted_objects = set()  # Set of highlighted object IDs
        
        # Visualization settings
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Filters
        self.visible_object_types = set()  # Set of visible object types
        self.visible_objects = set()  # Set of visible object IDs
        self.filter_confidence = 0.5  # Minimum confidence for visible objects
        
        # Query engine
        self.query_engine = None
        self.last_query_result = None
        
        # File paths
        self.current_pointcloud_path = None
        self.current_mesh_path = None
        self.current_scene_graph_path = None
        
        # Open3D visualization objects
        self.vis = None  # Open3D visualizer
        self.vis_geometry = []  # Geometries added to visualizer


class Open3DVisualizerWidget(QWidget):
    """Widget embedding an Open3D visualizer."""
    
    def __init__(self, parent=None):
        """Initialize Open3D visualizer widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Create Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        
        # Get the window handle
        vis_window = self.vis.get_render_window()
        
        if using_pyqt:
            from PyQt5.QtWidgets import QWidget
            # Create Qt widget from window handle
            self.vis_widget = QWidget.createWindowContainer(vis_window, self)
        else:
            from PySide2.QtWidgets import QWidget
            # Create Qt widget from window handle
            self.vis_widget = QWidget.createWindowContainer(vis_window, self)
        
        # Add to layout
        self.layout.addWidget(self.vis_widget)
        
        # Get render options
        self.render_options = self.vis.get_render_option()
        
        # Set default render options
        self.render_options.background_color = np.array(DEFAULT_SETTINGS['background_color'])
        self.render_options.point_size = DEFAULT_SETTINGS['point_size']
        
        # Get view control
        self.view_control = self.vis.get_view_control()
        
        # Register key callbacks
        self.vis.register_key_callback(ord('R'), self._reset_view)
    
    def _reset_view(self, vis):
        """Reset view to default."""
        self.view_control.reset_camera_to_default()
        return True
    
    def update_render_options(self, settings):
        """Update render options from settings.
        
        Args:
            settings: Rendering settings
        """
        self.render_options.background_color = np.array(settings['background_color'])
        self.render_options.point_size = settings['point_size']
        self.render_options.show_coordinate_frame = settings['show_axes']
        
        # Update the view
        self.vis.update_renderer()


class InteractiveViewer(QMainWindow):
    """Interactive viewer for semantically-enriched 3D scenes."""
    
    def __init__(self, 
                 pointcloud_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 scene_graph_path: Optional[str] = None):
        """Initialize interactive viewer.
        
        Args:
            pointcloud_path: Optional path to point cloud file
            mesh_path: Optional path to mesh file
            scene_graph_path: Optional path to scene graph file
        """
        super().__init__()
        
        # Initialize state
        self.state = ViewerState()
        
        # Set up UI
        self._init_ui()
        
        # Load files if provided
        if pointcloud_path and os.path.exists(pointcloud_path):
            self._load_pointcloud(pointcloud_path)
        
        if mesh_path and os.path.exists(mesh_path):
            self._load_mesh(mesh_path)
        
        if scene_graph_path and os.path.exists(scene_graph_path):
            self._load_scene_graph(scene_graph_path)
        
        # Initialize query engine in background
        self._init_query_engine_async()
    
    def _init_ui(self):
        """Initialize user interface."""
        # Main window properties
        self.setWindowTitle("RECONTEXT: 3D Scene Understanding")
        self.resize(1200, 800)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Main layout is a horizontal splitter
        self.main_layout = QHBoxLayout(self.main_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Left panel for controls
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Right panel for visualization
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Add panels to splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        
        # Set initial sizes (30% left, 70% right)
        self.splitter.setSizes([300, 700])
        
        # Create visualization widget
        self.vis_widget = Open3DVisualizerWidget()
        self.right_layout.addWidget(self.vis_widget)
        
        # Get visualizer from widget
        self.state.vis = self.vis_widget.vis
        
        # Create tab widget for left panel
        self.tab_widget = QTabWidget()
        self.left_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_data_tab()
        self._create_objects_tab()
        self._create_query_tab()
        self._create_settings_tab()
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
    
    def _create_menu_bar(self):
        """Create menu bar."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Open point cloud
        open_pcd_action = QAction("Open Point Cloud", self)
        open_pcd_action.triggered.connect(self._on_open_pointcloud)
        file_menu.addAction(open_pcd_action)
        
        # Open mesh
        open_mesh_action = QAction("Open Mesh", self)
        open_mesh_action.triggered.connect(self._on_open_mesh)
        file_menu.addAction(open_mesh_action)
        
        # Open scene graph
        open_sg_action = QAction("Open Scene Graph", self)
        open_sg_action.triggered.connect(self._on_open_scene_graph)
        file_menu.addAction(open_sg_action)
        
        file_menu.addSeparator()
        
        # Export scene graph
        export_sg_action = QAction("Export Scene Graph", self)
        export_sg_action.triggered.connect(self._on_export_scene_graph)
        file_menu.addAction(export_sg_action)
        
        # Export visualization
        export_vis_action = QAction("Export Visualization", self)
        export_vis_action.triggered.connect(self._on_export_visualization)
        file_menu.addAction(export_vis_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Reset view
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self._on_reset_view)
        view_menu.addAction(reset_view_action)
        
        view_menu.addSeparator()
        
        # Show/hide options
        show_labels_action = QAction("Show Labels", self)
        show_labels_action.setCheckable(True)
        show_labels_action.setChecked(self.state.settings['show_labels'])
        show_labels_action.triggered.connect(self._on_toggle_labels)
        view_menu.addAction(show_labels_action)
        
        show_bbox_action = QAction("Show Bounding Boxes", self)
        show_bbox_action.setCheckable(True)
        show_bbox_action.setChecked(self.state.settings['show_bounding_boxes'])
        show_bbox_action.triggered.connect(self._on_toggle_bounding_boxes)
        view_menu.addAction(show_bbox_action)
        
        show_axes_action = QAction("Show Axes", self)
        show_axes_action.setCheckable(True)
        show_axes_action.setChecked(self.state.settings['show_axes'])
        show_axes_action.triggered.connect(self._on_toggle_axes)
        view_menu.addAction(show_axes_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
        # Keyboard shortcuts
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._on_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def _create_toolbar(self):
        """Create toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)
        
        # Add actions
        self.toolbar.addAction(QAction("Reset View", self, triggered=self._on_reset_view))
        
        self.toolbar.addSeparator()
        
        # Add query box
        self.toolbar.addWidget(QLabel("Query:"))
        self.quick_query_box = QLineEdit()
        self.quick_query_box.setPlaceholderText("Ask a question about the scene...")
        self.quick_query_box.setMinimumWidth(300)
        self.quick_query_box.returnPressed.connect(self._on_quick_query)
        self.toolbar.addWidget(self.quick_query_box)
        
        query_button = QPushButton("Ask")
        query_button.clicked.connect(self._on_quick_query)
        self.toolbar.addWidget(query_button)
    
    def _create_data_tab(self):
        """Create data tab for loading and viewing data."""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # Data sources group
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout(data_group)
        
        # Point cloud
        pcd_layout = QHBoxLayout()
        pcd_layout.addWidget(QLabel("Point Cloud:"))
        self.pcd_path_label = QLabel("None")
        pcd_layout.addWidget(self.pcd_path_label, 1)
        pcd_load_button = QPushButton("Load")
        pcd_load_button.clicked.connect(self._on_open_pointcloud)
        pcd_layout.addWidget(pcd_load_button)
        data_layout.addLayout(pcd_layout)
        
        # Mesh
        mesh_layout = QHBoxLayout()
        mesh_layout.addWidget(QLabel("Mesh:"))
        self.mesh_path_label = QLabel("None")
        mesh_layout.addWidget(self.mesh_path_label, 1)
        mesh_load_button = QPushButton("Load")
        mesh_load_button.clicked.connect(self._on_open_mesh)
        mesh_layout.addWidget(mesh_load_button)
        data_layout.addLayout(mesh_layout)
        
        # Scene graph
        sg_layout = QHBoxLayout()
        sg_layout.addWidget(QLabel("Scene Graph:"))
        self.sg_path_label = QLabel("None")
        sg_layout.addWidget(self.sg_path_label, 1)
        sg_load_button = QPushButton("Load")
        sg_load_button.clicked.connect(self._on_open_scene_graph)
        sg_layout.addWidget(sg_load_button)
        data_layout.addLayout(sg_layout)
        
        layout.addWidget(data_group)
        
        # Scene info group
        info_group = QGroupBox("Scene Information")
        info_layout = QVBoxLayout(info_group)
        
        # Stats text area
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumHeight(200)
        info_layout.addWidget(self.stats_text)
        
        layout.addWidget(info_group)
        
        # Controls group
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Display options
        display_layout = QVBoxLayout()
        
        # Point size slider
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(int(self.state.settings['point_size']))
        self.point_size_slider.valueChanged.connect(self._on_point_size_changed)
        point_size_layout.addWidget(self.point_size_slider)
        display_layout.addLayout(point_size_layout)
        
        # Checkboxes for display options
        self.show_labels_checkbox = QCheckBox("Show Labels")
        self.show_labels_checkbox.setChecked(self.state.settings['show_labels'])
        self.show_labels_checkbox.stateChanged.connect(self._on_toggle_labels)
        display_layout.addWidget(self.show_labels_checkbox)
        
        self.show_bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.show_bbox_checkbox.setChecked(self.state.settings['show_bounding_boxes'])
        self.show_bbox_checkbox.stateChanged.connect(self._on_toggle_bounding_boxes)
        display_layout.addWidget(self.show_bbox_checkbox)
        
        self.show_axes_checkbox = QCheckBox("Show Axes")
        self.show_axes_checkbox.setChecked(self.state.settings['show_axes'])
        self.show_axes_checkbox.stateChanged.connect(self._on_toggle_axes)
        display_layout.addWidget(self.show_axes_checkbox)
        
        controls_layout.addLayout(display_layout)
        
        # View controls
        view_layout = QHBoxLayout()
        reset_view_button = QPushButton("Reset View")
        reset_view_button.clicked.connect(self._on_reset_view)
        view_layout.addWidget(reset_view_button)
        
        controls_layout.addLayout(view_layout)
        
        layout.addWidget(controls_group)
        
        # Add tab
        self.tab_widget.addTab(data_tab, "Data")
    
    def _create_objects_tab(self):
        """Create objects tab for viewing and filtering objects."""
        objects_tab = QWidget()
        layout = QVBoxLayout(objects_tab)
        
        # Objects list group
        objects_group = QGroupBox("Scene Objects")
        objects_layout = QVBoxLayout(objects_group)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.object_filter_box = QLineEdit()
        self.object_filter_box.setPlaceholderText("Filter by name...")
        self.object_filter_box.textChanged.connect(self._on_object_filter_changed)
        filter_layout.addWidget(self.object_filter_box)
        objects_layout.addLayout(filter_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Min Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.state.filter_confidence * 100))
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel(f"{self.state.filter_confidence:.2f}")
        conf_layout.addWidget(self.conf_value_label)
        objects_layout.addLayout(conf_layout)
        
        # Objects list
        self.objects_list = QListWidget()
        self.objects_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.objects_list.itemSelectionChanged.connect(self._on_object_selection_changed)
        objects_layout.addWidget(self.objects_list)
        
        # Object controls
        obj_controls_layout = QHBoxLayout()
        
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self._on_select_all_objects)
        obj_controls_layout.addWidget(select_all_button)
        
        clear_selection_button = QPushButton("Clear Selection")
        clear_selection_button.clicked.connect(self._on_clear_object_selection)
        obj_controls_layout.addWidget(clear_selection_button)
        
        objects_layout.addLayout(obj_controls_layout)
        
        layout.addWidget(objects_group)
        
        # Object details group
        details_group = QGroupBox("Object Details")
        details_layout = QVBoxLayout(details_group)
        
        # Details text area
        self.object_details_text = QTextEdit()
        self.object_details_text.setReadOnly(True)
        details_layout.addWidget(self.object_details_text)
        
        layout.addWidget(details_group)
        
        # Relationships group
        relations_group = QGroupBox("Object Relationships")
        relations_layout = QVBoxLayout(relations_group)
        
        # Relationships list
        self.relationships_list = QListWidget()
        self.relationships_list.itemClicked.connect(self._on_relationship_clicked)
        relations_layout.addWidget(self.relationships_list)
        
        layout.addWidget(relations_group)
        
        # Add tab
        self.tab_widget.addTab(objects_tab, "Objects")
    
    def _create_query_tab(self):
        """Create query tab for natural language interaction."""
        query_tab = QWidget()
        layout = QVBoxLayout(query_tab)
        
        # Query input group
        query_group = QGroupBox("Natural Language Query")
        query_layout = QVBoxLayout(query_group)
        
        # Query input box
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about the scene...")
        self.query_input.returnPressed.connect(self._on_query)
        query_layout.addWidget(self.query_input)
        
        # Query button
        query_button = QPushButton("Ask")
        query_button.clicked.connect(self._on_query)
        query_layout.addWidget(query_button)
        
        # Loading indicator
        self.query_loading_label = QLabel("Initializing query engine...")
        query_layout.addWidget(self.query_loading_label)
        
        layout.addWidget(query_group)
        
        # Query result group
        result_group = QGroupBox("Query Results")
        result_layout = QVBoxLayout(result_group)
        
        # Answer text area
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setMinimumHeight(100)
        result_layout.addWidget(self.answer_text)
        
        # Results list
        self.query_results_list = QListWidget()
        self.query_results_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.query_results_list.itemSelectionChanged.connect(self._on_query_result_selection_changed)
        result_layout.addWidget(self.query_results_list)
        
        # Highlight button
        highlight_button = QPushButton("Highlight Results")
        highlight_button.clicked.connect(self._on_highlight_query_results)
        result_layout.addWidget(highlight_button)
        
        layout.addWidget(result_group)
        
        # Query history group
        history_group = QGroupBox("Query History")
        history_layout = QVBoxLayout(history_group)
        
        # History list
        self.query_history_list = QListWidget()
        self.query_history_list.itemClicked.connect(self._on_query_history_clicked)
        history_layout.addWidget(self.query_history_list)
        
        layout.addWidget(history_group)
        
        # Add tab
        self.tab_widget.addTab(query_tab, "Query")
    
    def _create_settings_tab(self):
        """Create settings tab for application settings."""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        
        # Display settings group
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Background color
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background Color:"))
        
        # Color buttons
        color_buttons_layout = QHBoxLayout()
        
        dark_button = QPushButton("Dark")
        dark_button.clicked.connect(lambda: self._on_set_background_color([0.1, 0.1, 0.1]))
        color_buttons_layout.addWidget(dark_button)
        
        light_button = QPushButton("Light")
        light_button.clicked.connect(lambda: self._on_set_background_color([0.9, 0.9, 0.9]))
        color_buttons_layout.addWidget(light_button)
        
        blue_button = QPushButton("Blue")
        blue_button.clicked.connect(lambda: self._on_set_background_color([0.1, 0.1, 0.3]))
        color_buttons_layout.addWidget(blue_button)
        
        custom_button = QPushButton("Custom...")
        custom_button.clicked.connect(self._on_custom_background_color)
        color_buttons_layout.addWidget(custom_button)
        
        bg_layout.addLayout(color_buttons_layout)
        display_layout.addLayout(bg_layout)
        
        # Label size
        label_size_layout = QHBoxLayout()
        label_size_layout.addWidget(QLabel("Label Size:"))
        self.label_size_slider = QSlider(Qt.Horizontal)
        self.label_size_slider.setMinimum(8)
        self.label_size_slider.setMaximum(32)
        self.label_size_slider.setValue(self.state.settings['label_size'])
        self.label_size_slider.valueChanged.connect(self._on_label_size_changed)
        label_size_layout.addWidget(self.label_size_slider)
        self.label_size_value = QLabel(str(self.state.settings['label_size']))
        label_size_layout.addWidget(self.label_size_value)
        display_layout.addLayout(label_size_layout)
        
        # Highlight color
        highlight_layout = QHBoxLayout()
        highlight_layout.addWidget(QLabel("Highlight Color:"))
        
        # Color buttons
        highlight_color_layout = QHBoxLayout()
        
        orange_button = QPushButton("Orange")
        orange_button.clicked.connect(lambda: self._on_set_highlight_color([1.0, 0.5, 0.0]))
        highlight_color_layout.addWidget(orange_button)
        
        red_button = QPushButton("Red")
        red_button.clicked.connect(lambda: self._on_set_highlight_color([1.0, 0.0, 0.0]))
        highlight_color_layout.addWidget(red_button)
        
        green_button = QPushButton("Green")
        green_button.clicked.connect(lambda: self._on_set_highlight_color([0.0, 1.0, 0.0]))
        highlight_color_layout.addWidget(green_button)
        
        custom_hl_button = QPushButton("Custom...")
        custom_hl_button.clicked.connect(self._on_custom_highlight_color)
        highlight_color_layout.addWidget(custom_hl_button)
        
        highlight_layout.addLayout(highlight_color_layout)
        display_layout.addLayout(highlight_layout)
        
        layout.addWidget(display_group)
        
        # Query engine settings group
        query_group = QGroupBox("Query Engine Settings")
        query_layout = QVBoxLayout(query_group)
        
        # Model type
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["default", "large"])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        query_layout.addLayout(model_layout)
        
        # Initialize button
        init_engine_button = QPushButton("Initialize Query Engine")
        init_engine_button.clicked.connect(self._init_query_engine_async)
        query_layout.addWidget(init_engine_button)
        
        layout.addWidget(query_group)
        
        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        # Export buttons
        export_sg_button = QPushButton("Export Scene Graph")
        export_sg_button.clicked.connect(self._on_export_scene_graph)
        export_layout.addWidget(export_sg_button)
        
        export_vis_button = QPushButton("Export Visualization")
        export_vis_button.clicked.connect(self._on_export_visualization)
        export_layout.addWidget(export_vis_button)
        
        layout.addWidget(export_group)
        
        # Add tab
        self.tab_widget.addTab(settings_tab, "Settings")
    
    def _load_pointcloud(self, filepath: str):
        """Load point cloud from file.
        
        Args:
            filepath: Path to point cloud file
        """
        try:
            # Load point cloud
            pointcloud = o3d.io.read_point_cloud(filepath)
            
            if not pointcloud.has_points():
                raise ValueError("Point cloud has no points")
            
            # Store in state
            self.state.pointcloud = pointcloud
            self.state.current_pointcloud_path = filepath
            
            # Update path label
            self.pcd_path_label.setText(os.path.basename(filepath))
            
            # Visualize
            self._visualize_pointcloud()
            
            # Update status
            self.status_bar.showMessage(f"Loaded point cloud from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading point cloud: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load point cloud: {str(e)}")
    
    def _load_mesh(self, filepath: str):
        """Load mesh from file.
        
        Args:
            filepath: Path to mesh file
        """
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(filepath)
            
            if not mesh.has_triangles():
                raise ValueError("Mesh has no triangles")
            
            # Store in state
            self.state.mesh = mesh
            self.state.current_mesh_path = filepath
            
            # Update path label
            self.mesh_path_label.setText(os.path.basename(filepath))
            
            # Visualize
            self._visualize_mesh()
            
            # Update status
            self.status_bar.showMessage(f"Loaded mesh from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading mesh: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load mesh: {str(e)}")
    
    def _load_scene_graph(self, filepath: str):
        """Load scene graph from file.
        
        Args:
            filepath: Path to scene graph file
        """
        try:
            # Load scene graph
            scene_graph = SceneGraph.load(filepath)
            
            # Store in state
            self.state.scene_graph = scene_graph
            self.state.current_scene_graph_path = filepath
            
            # Update path label
            self.sg_path_label.setText(os.path.basename(filepath))
            
            # Update scene information
            self._update_scene_info()
            
            # Update object list
            self._update_object_list()
            
            # Visualize
            self._visualize_scene_graph()
            
            # Update status
            self.status_bar.showMessage(f"Loaded scene graph from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading scene graph: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load scene graph: {str(e)}")
    
    def _visualize_pointcloud(self):
        """Visualize point cloud."""
        if self.state.pointcloud is None:
            return
            
        # Clear previous geometries
        self.state.vis.clear_geometries()
        self.state.vis_geometry = []
        
        # Add point cloud
        self.state.vis.add_geometry(self.state.pointcloud)
        self.state.vis_geometry.append(self.state.pointcloud)
        
        # Update view
        self.state.vis.update_renderer()
        self.state.vis.reset_view_point(True)
    
    def _visualize_mesh(self):
        """Visualize mesh."""
        if self.state.mesh is None:
            return
            
        # Clear previous geometries
        self.state.vis.clear_geometries()
        self.state.vis_geometry = []
        
        # Add mesh
        self.state.vis.add_geometry(self.state.mesh)
        self.state.vis_geometry.append(self.state.mesh)
        
        # Update view
        self.state.vis.update_renderer()
        self.state.vis.reset_view_point(True)
    
    def _visualize_scene_graph(self):
        """Visualize scene graph."""
        if self.state.scene_graph is None:
            return
            
        # Clear previous visualization objects
        self.state.vis_objects = {}
        
        # Clear previous geometries
        self.state.vis.clear_geometries()
        self.state.vis_geometry = []
        
        # Process each object in scene graph
        for obj_id, obj in self.state.scene_graph.objects.items():
            # Create visualization for this object
            vis_obj = self._create_object_visualization(obj)
            self.state.vis_objects[obj_id] = vis_obj
            
            # Add geometries to visualizer
            for geom in vis_obj['geometries']:
                self.state.vis.add_geometry(geom)
                self.state.vis_geometry.append(geom)
        
        # Add original point cloud or mesh if available
        if self.state.pointcloud is not None:
            # Use very small point size for original cloud
            orig_cloud = o3d.geometry.PointCloud(self.state.pointcloud)
            orig_cloud.paint_uniform_color([0.7, 0.7, 0.7])
            
            self.state.vis.add_geometry(orig_cloud)
            self.state.vis_geometry.append(orig_cloud)
        
        if self.state.mesh is not None:
            # Use transparent material for original mesh
            orig_mesh = o3d.geometry.TriangleMesh(self.state.mesh)
            if not orig_mesh.has_vertex_colors():
                orig_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
            self.state.vis.add_geometry(orig_mesh)
            self.state.vis_geometry.append(orig_mesh)
        
        # Update all visible objects
        self._update_visible_objects()
        
        # Update view
        self.state.vis.update_renderer()
        self.state.vis.reset_view_point(True)
    
    def _create_object_visualization(self, obj: Object3D) -> Dict:
        """Create visualization objects for a semantic object.
        
        Args:
            obj: 3D object
            
        Returns:
            Dictionary of visualization objects
        """
        vis_obj = {
            'geometries': [],
            'bbox': None,
            'label': None,
            'point_cloud': None,
            'visible': True
        }
        
        # Create point cloud for the object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj.points)
        
        # Set color
        color = np.array(obj.color) / 255.0
        point_cloud.paint_uniform_color(color)
        
        vis_obj['point_cloud'] = point_cloud
        vis_obj['geometries'].append(point_cloud)
        
        # Create bounding box
        min_bound = obj.bbox[:3]
        max_bound = obj.bbox[3:]
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = np.array([0.0, 0.0, 0.0])  # Black wireframe
        
        vis_obj['bbox'] = bbox
        
        if self.state.settings['show_bounding_boxes']:
            vis_obj['geometries'].append(bbox)
        
        # Create text label (this is not directly supported in Open3D)
        # For a proper implementation, you'd need to use custom rendering
        # or use a GUI library that supports 3D text
        
        return vis_obj
    
    def _update_scene_info(self):
        """Update scene information display."""
        if self.state.scene_graph is None:
            self.stats_text.setText("No scene graph loaded")
            return
        
        scene_info = self.state.scene_graph.scene_info
        stats = scene_info.get('stats', {})
        
        # Create formatted text
        info_text = f"Scene: {scene_info.get('name', 'Unnamed')}\n\n"
        
        # Dimensions
        dims = scene_info.get('dimensions')
        if dims is not None:
            info_text += f"Dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} units\n\n"
        
        # Object counts
        info_text += f"Objects: {stats.get('object_count', 0)}\n"
        info_text += f"Relationships: {stats.get('relationship_count', 0)}\n\n"
        
        # Object types
        object_types = stats.get('object_types', {})
        if object_types:
            info_text += "Object Types:\n"
            for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True):
                info_text += f"  {obj_type}: {count}\n"
            info_text += "\n"
        
        # Relationship types
        rel_types = stats.get('relationship_types', {})
        if rel_types:
            info_text += "Relationship Types:\n"
            for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                info_text += f"  {rel_type}: {count}\n"
            info_text += "\n"
        
        self.stats_text.setText(info_text)
    
    def _update_object_list(self):
        """Update object list with current scene graph objects."""
        self.objects_list.clear()
        
        if self.state.scene_graph is None:
            return
        
        # Add items for each object
        for obj_id, obj in sorted(self.state.scene_graph.objects.items(), 
                                key=lambda x: x[1].label):
            # Create item text
            item_text = f"{obj.label} (ID: {obj_id})"
            
            # Add to list
            self.objects_list.addItem(item_text)
    
    def _update_visible_objects(self):
        """Update visibility of objects based on current filters."""
        if self.state.scene_graph is None:
            return
        
        # Get filter text
        filter_text = self.object_filter_box.text().lower()
        
        # Calculate visible objects
        visible_objects = set()
        
        for obj_id, obj in self.state.scene_graph.objects.items():
            # Check filter
            if filter_text and filter_text not in obj.label.lower():
                continue
                
            # Object passes filters
            visible_objects.add(obj_id)
        
        # Update state
        self.state.visible_objects = visible_objects
        
        # Update visualization
        for obj_id, vis_obj in self.state.vis_objects.items():
            visible = obj_id in visible_objects
            vis_obj['visible'] = visible
            
            # TODO: Open3D doesn't directly support changing visibility
            # Instead, we'd need to remove/add geometries
    
    def _update_object_details(self, obj_id: int):
        """Update object details display.
        
        Args:
            obj_id: Object ID
        """
        if self.state.scene_graph is None:
            return
            
        obj = self.state.scene_graph.get_object(obj_id)
        if obj is None:
            self.object_details_text.setText("")
            return
        
        # Create formatted text
        details_text = f"Object ID: {obj.id}\n"
        details_text += f"Label: {obj.label}\n"
        details_text += f"Class ID: {obj.class_id}\n\n"
        
        # Position and size
        details_text += f"Center: [{obj.center[0]:.2f}, {obj.center[1]:.2f}, {obj.center[2]:.2f}]\n"
        details_text += f"Size: [{obj.size[0]:.2f}, {obj.size[1]:.2f}, {obj.size[2]:.2f}]\n"
        details_text += f"Volume: {obj.volume:.2f} cubic units\n"
        details_text += f"Points: {obj.point_count}\n\n"
        
        # Color
        details_text += f"Color: RGB({obj.color[0]}, {obj.color[1]}, {obj.color[2]})\n\n"
        
        # Attributes
        if obj.attributes:
            details_text += "Attributes:\n"
            for attr, value in obj.attributes.items():
                details_text += f"  {attr}: {value}\n"
        
        self.object_details_text.setText(details_text)
    
    def _update_relationships_list(self, obj_id: int):
        """Update relationships list for an object.
        
        Args:
            obj_id: Object ID
        """
        self.relationships_list.clear()
        
        if self.state.scene_graph is None:
            return
            
        # Get object
        obj = self.state.scene_graph.get_object(obj_id)
        if obj is None:
            return
        
        # Get relationships
        relationships = []
        
        # Outgoing relationships (obj -> other)
        for rel_id, rel in self.state.scene_graph.relationships.items():
            if rel.source_id == obj_id:
                target = self.state.scene_graph.get_object(rel.target_id)
                if target:
                    relationships.append((rel_id, f"{obj.label} {rel.type} {target.label}"))
        
        # Incoming relationships (other -> obj)
        for rel_id, rel in self.state.scene_graph.relationships.items():
            if rel.target_id == obj_id:
                source = self.state.scene_graph.get_object(rel.source_id)
                if source:
                    relationships.append((rel_id, f"{source.label} {rel.type} {obj.label}"))
        
        # Sort by relationship text
        relationships.sort(key=lambda x: x[1])
        
        # Add to list
        for rel_id, rel_text in relationships:
            self.relationships_list.addItem(rel_text)
            
            # Store relationship ID as item data
            item = self.relationships_list.item(self.relationships_list.count() - 1)
            if using_pyqt:
                item.setData(Qt.UserRole, rel_id)
            else:
                item.setData(Qt.UserRole, rel_id)
    
    def _highlight_selected_objects(self):
        """Highlight selected objects in visualization."""
        if not self.state.settings['highlight_selected']:
            return
            
        # Reset highlights
        self._clear_highlights()
        
        # Set selected objects as highlighted
        self.state.highlighted_objects = self.state.selected_objects.copy()
        
        # Apply highlights
        self._apply_highlights()
    
    def _highlight_query_results(self):
        """Highlight objects from query results."""
        if self.state.last_query_result is None:
            return
            
        # Clear highlights
        self._clear_highlights()
        
        # Get objects from query result
        objects = self.state.last_query_result.objects
        
        # Set as highlighted
        self.state.highlighted_objects = set(objects)
        
        # Apply highlights
        self._apply_highlights()
    
    def _clear_highlights(self):
        """Clear all object highlights."""
        # Reset all point cloud colors
        for obj_id, vis_obj in self.state.vis_objects.items():
            if vis_obj['point_cloud'] is not None:
                obj = self.state.scene_graph.get_object(obj_id)
                if obj is not None:
                    # Reset to original color
                    color = np.array(obj.color) / 255.0
                    vis_obj['point_cloud'].paint_uniform_color(color)
        
        # Update visualization
        self.state.vis.update_renderer()
    
    def _apply_highlights(self):
        """Apply highlights to highlighted objects."""
        # Highlight selected objects
        highlight_color = self.state.settings['highlight_color']
        
        for obj_id in self.state.highlighted_objects:
            vis_obj = self.state.vis_objects.get(obj_id)
            if vis_obj is not None and vis_obj['point_cloud'] is not None:
                # Apply highlight color
                vis_obj['point_cloud'].paint_uniform_color(highlight_color)
        
        # Update visualization
        self.state.vis.update_renderer()
    
    def _init_query_engine_async(self):
        """Initialize query engine asynchronously."""
        # Show loading message
        self.query_loading_label.setText("Initializing query engine...")
        self.query_loading_label.setVisible(True)
        
        # Get model type
        model_type = self.model_type_combo.currentText()
        
        # Create async task
        def init_engine():
            try:
                engine = QueryEngine(model_type=model_type)
                return engine
            except Exception as e:
                logger.error(f"Error initializing query engine: {str(e)}")
                raise
        
        task = AsyncTask(init_engine)
        
        # Connect signals
        task.finished.connect(self._on_query_engine_initialized)
        task.error.connect(self._on_query_engine_error)
        
        # Start task
        task.start()
    
    def _process_query_async(self, query: str):
        """Process natural language query asynchronously.
        
        Args:
            query: Query string
        """
        if self.state.query_engine is None or self.state.scene_graph is None:
            QMessageBox.warning(self, "Warning", "Query engine or scene graph not initialized")
            return
            
        # Show loading message
        self.query_loading_label.setText("Processing query...")
        self.query_loading_label.setVisible(True)
        
        # Create async task
        def process_query():
            try:
                result = self.state.query_engine.query(self.state.scene_graph, query)
                return result
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                raise
        
        task = AsyncTask(process_query)
        
        # Connect signals
        task.finished.connect(self._on_query_result)
        task.error.connect(self._on_query_error)
        
        # Start task
        task.start()
    
    # Event handlers
    
    def _on_open_pointcloud(self):
        """Handle open point cloud action."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "", "Point Cloud Files (*.ply *.pcd);;All Files (*)")
            
        if filepath:
            self._load_pointcloud(filepath)
    
    def _on_open_mesh(self):
        """Handle open mesh action."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh", "", "Mesh Files (*.ply *.obj *.off);;All Files (*)")
            
        if filepath:
            self._load_mesh(filepath)
    
    def _on_open_scene_graph(self):
        """Handle open scene graph action."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Scene Graph", "", "Scene Graph Files (*.pkl);;All Files (*)")
            
        if filepath:
            self._load_scene_graph(filepath)
    
    def _on_export_scene_graph(self):
        """Handle export scene graph action."""
        if self.state.scene_graph is None:
            QMessageBox.warning(self, "Warning", "No scene graph to export")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Scene Graph", "", "Scene Graph Files (*.pkl);;All Files (*)")
            
        if filepath:
            try:
                self.state.scene_graph.save(filepath)
                QMessageBox.information(self, "Success", f"Scene graph exported to {filepath}")
            except Exception as e:
                logger.error(f"Error exporting scene graph: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to export scene graph: {str(e)}")
    
    def _on_export_visualization(self):
        """Handle export visualization action."""
        if not self.state.vis_geometry:
            QMessageBox.warning(self, "Warning", "No visualization to export")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "", "Image Files (*.png *.jpg);;All Files (*)")
            
        if filepath:
            try:
                # Capture screenshot
                self.state.vis.capture_screen_image(filepath, True)
                QMessageBox.information(self, "Success", f"Visualization exported to {filepath}")
            except Exception as e:
                logger.error(f"Error exporting visualization: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to export visualization: {str(e)}")
    
    def _on_reset_view(self):
        """Handle reset view action."""
        if self.state.vis is not None:
            self.state.vis.reset_view_point(True)
    
    def _on_toggle_labels(self, state):
        """Handle toggle labels action.
        
        Args:
            state: Checkbox state
        """
        checked = bool(state)
        self.state.settings['show_labels'] = checked
        self.show_labels_checkbox.setChecked(checked)
        
        # TODO: Update visualization
    
    def _on_toggle_bounding_boxes(self, state):
        """Handle toggle bounding boxes action.
        
        Args:
            state: Checkbox state
        """
        checked = bool(state)
        self.state.settings['show_bounding_boxes'] = checked
        self.show_bbox_checkbox.setChecked(checked)
        
        # Update bounding boxes in visualization
        for obj_id, vis_obj in self.state.vis_objects.items():
            if vis_obj['bbox'] is not None:
                if checked:
                    # Add bbox if not already added
                    if vis_obj['bbox'] not in self.state.vis_geometry:
                        self.state.vis.add_geometry(vis_obj['bbox'])
                        self.state.vis_geometry.append(vis_obj['bbox'])
                else:
                    # Remove bbox if present
                    if vis_obj['bbox'] in self.state.vis_geometry:
                        self.state.vis.remove_geometry(vis_obj['bbox'])
                        self.state.vis_geometry.remove(vis_obj['bbox'])
        
        # Update visualization
        self.state.vis.update_renderer()
    
    def _on_toggle_axes(self, state):
        """Handle toggle axes action.
        
        Args:
            state: Checkbox state
        """
        checked = bool(state)
        self.state.settings['show_axes'] = checked
        self.show_axes_checkbox.setChecked(checked)
        
        # Update render options
        self.vis_widget.render_options.show_coordinate_frame = checked
        self.state.vis.update_renderer()
    
    def _on_point_size_changed(self, value):
        """Handle point size slider change.
        
        Args:
            value: Slider value
        """
        self.state.settings['point_size'] = float(value)
        
        # Update render options
        self.vis_widget.render_options.point_size = float(value)
        self.state.vis.update_renderer()
    
    def _on_label_size_changed(self, value):
        """Handle label size slider change.
        
        Args:
            value: Slider value
        """
        self.state.settings['label_size'] = value
        self.label_size_value.setText(str(value))
        
        # TODO: Update label size in visualization
    
    def _on_set_background_color(self, color):
        """Handle background color change.
        
        Args:
            color: RGB color [r, g, b]
        """
        self.state.settings['background_color'] = color
        
        # Update render options
        self.vis_widget.render_options.background_color = np.array(color)
        self.state.vis.update_renderer()
    
    def _on_custom_background_color(self):
        """Handle custom background color selection."""
        if using_pyqt:
            from PyQt5.QtWidgets import QColorDialog
            
            # Convert current color to Qt color
            current = self.state.settings['background_color']
            qt_color = QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
            
            # Open color dialog
            color = QColorDialog.getColor(qt_color, self, "Select Background Color")
            
            if color.isValid():
                new_color = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
                self._on_set_background_color(new_color)
        else:
            from PySide2.QtWidgets import QColorDialog
            
            # Convert current color to Qt color
            current = self.state.settings['background_color']
            qt_color = QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
            
            # Open color dialog
            color = QColorDialog.getColor(qt_color, self, "Select Background Color")
            
            if color.isValid():
                new_color = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
                self._on_set_background_color(new_color)
    
    def _on_set_highlight_color(self, color):
        """Handle highlight color change.
        
        Args:
            color: RGB color [r, g, b]
        """
        self.state.settings['highlight_color'] = color
        
        # Update current highlights
        self._apply_highlights()
    
    def _on_custom_highlight_color(self):
        """Handle custom highlight color selection."""
        if using_pyqt:
            from PyQt5.QtWidgets import QColorDialog
            
            # Convert current color to Qt color
            current = self.state.settings['highlight_color']
            qt_color = QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
            
            # Open color dialog
            color = QColorDialog.getColor(qt_color, self, "Select Highlight Color")
            
            if color.isValid():
                new_color = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
                self._on_set_highlight_color(new_color)
        else:
            from PySide2.QtWidgets import QColorDialog
            
            # Convert current color to Qt color
            current = self.state.settings['highlight_color']
            qt_color = QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
            
            # Open color dialog
            color = QColorDialog.getColor(qt_color, self, "Select Highlight Color")
            
            if color.isValid():
                new_color = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
                self._on_set_highlight_color(new_color)
    
    def _on_object_filter_changed(self, text):
        """Handle object filter text change.
        
        Args:
            text: Filter text
        """
        # Update visible objects
        self._update_visible_objects()
    
    def _on_confidence_changed(self, value):
        """Handle confidence threshold slider change.
        
        Args:
            value: Slider value
        """
        # Update confidence threshold
        self.state.filter_confidence = value / 100.0
        self.conf_value_label.setText(f"{self.state.filter_confidence:.2f}")
        
        # Update visible objects
        self._update_visible_objects()
    
    def _on_object_selection_changed(self):
        """Handle object selection change."""
        selected_items = self.objects_list.selectedItems()
        
        # Clear previous selection
        self.state.selected_objects.clear()
        
        # Get object IDs from selected items
        for item in selected_items:
            text = item.text()
            # Extract ID from text (format: "Label (ID: X)")
            try:
                obj_id = int(text.split("ID: ")[1].rstrip(")"))
                self.state.selected_objects.add(obj_id)
            except (IndexError, ValueError):
                continue
        
        # Update details if single object selected
        if len(self.state.selected_objects) == 1:
            obj_id = next(iter(self.state.selected_objects))
            self._update_object_details(obj_id)
            self._update_relationships_list(obj_id)
        else:
            self.object_details_text.setText("")
            self.relationships_list.clear()
        
        # Highlight selected objects
        self._highlight_selected_objects()
    
    def _on_select_all_objects(self):
        """Handle select all objects button."""
        self.objects_list.selectAll()
    
    def _on_clear_object_selection(self):
        """Handle clear selection button."""
        self.objects_list.clearSelection()
    
    def _on_relationship_clicked(self, item):
        """Handle relationship item click.
        
        Args:
            item: Clicked item
        """
        if using_pyqt:
            rel_id = item.data(Qt.UserRole)
        else:
            rel_id = item.data(Qt.UserRole)
            
        if rel_id is None or self.state.scene_graph is None:
            return
            
        # Get relationship
        rel = self.state.scene_graph.get_relationship(rel_id)
        if rel is None:
            return
            
        # Select related objects
        self.state.selected_objects = {rel.source_id, rel.target_id}
        
        # Update object list selection
        self.objects_list.clearSelection()
        for i in range(self.objects_list.count()):
            item = self.objects_list.item(i)
            text = item.text()
            try:
                obj_id = int(text.split("ID: ")[1].rstrip(")"))
                if obj_id in self.state.selected_objects:
                    item.setSelected(True)
            except (IndexError, ValueError):
                continue
        
        # Highlight selected objects
        self._highlight_selected_objects()
    
    def _on_query(self):
        """Handle query button."""
        if self.state.query_engine is None:
            QMessageBox.warning(self, "Warning", "Query engine not initialized")
            return
            
        if self.state.scene_graph is None:
            QMessageBox.warning(self, "Warning", "No scene graph loaded")
            return
            
        # Get query text
        query = self.query_input.text().strip()
        if not query:
            return
            
        # Process query
        self._process_query_async(query)
    
    def _on_quick_query(self):
        """Handle query from toolbar."""
        if self.state.query_engine is None:
            QMessageBox.warning(self, "Warning", "Query engine not initialized")
            return
            
        if self.state.scene_graph is None:
            QMessageBox.warning(self, "Warning", "No scene graph loaded")
            return
            
        # Get query text
        query = self.quick_query_box.text().strip()
        if not query:
            return
            
        # Process query
        self._process_query_async(query)
        
        # Also update the query input in the query tab
        self.query_input.setText(query)
    
    def _on_query_engine_initialized(self, engine):
        """Handle query engine initialization completion.
        
        Args:
            engine: Initialized query engine
        """
        self.state.query_engine = engine
        self.query_loading_label.setText("Query engine ready")
        self.status_bar.showMessage("Query engine initialized")
        
        # Hide loading label after a short delay
        QThread.sleep(1)
        self.query_loading_label.setVisible(False)
    
    def _on_query_engine_error(self, error_msg):
        """Handle query engine initialization error.
        
        Args:
            error_msg: Error message
        """
        self.query_loading_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Failed to initialize query engine: {error_msg}")
    
    def _on_query_result(self, result):
        """Handle query result.
        
        Args:
            result: Query result
        """
        # Store result
        self.state.last_query_result = result
        
        # Hide loading label
        self.query_loading_label.setVisible(False)
        
        # Update answer text
        self.answer_text.setText(result.answer)
        
        # Update results list
        self.query_results_list.clear()
        
        # Add objects
        if result.objects:
            self.query_results_list.addItem(f"--- Objects ({len(result.objects)}) ---")
            
            for obj_id in result.objects:
                obj = self.state.scene_graph.get_object(obj_id)
                if obj:
                    self.query_results_list.addItem(f"{obj.label} (ID: {obj_id})")
        
        # Add relationships
        if result.relationships:
            self.query_results_list.addItem(f"--- Relationships ({len(result.relationships)}) ---")
            
            for rel_id in result.relationships:
                rel = self.state.scene_graph.get_relationship(rel_id)
                if rel:
                    source = self.state.scene_graph.get_object(rel.source_id)
                    target = self.state.scene_graph.get_object(rel.target_id)
                    if source and target:
                        self.query_results_list.addItem(
                            f"{source.label} {rel.type} {target.label} (ID: {rel_id})")
        
        # Add to history
        self.query_history_list.addItem(result.query)
        
        # Highlight results
        self._highlight_query_results()
    
    def _on_query_error(self, error_msg):
        """Handle query processing error.
        
        Args:
            error_msg: Error message
        """
        self.query_loading_label.setVisible(False)
        QMessageBox.critical(self, "Error", f"Failed to process query: {error_msg}")
    
    def _on_query_result_selection_changed(self):
        """Handle query result selection change."""
        selected_items = self.query_results_list.selectedItems()
        
        # Skip header items
        selected_items = [item for item in selected_items if "---" not in item.text()]
        
        if not selected_items:
            return
            
        # Clear current selection
        self.state.selected_objects.clear()
        
        # Process selected items
        for item in selected_items:
            text = item.text()
            
            # Extract ID from text
            try:
                if "ID:" in text:
                    id_part = text.split("ID:")[1].strip()
                    obj_id = int(id_part.rstrip(")"))
                    
                    if "Relationship" in text:
                        # For relationships, select both source and target
                        rel = self.state.scene_graph.get_relationship(obj_id)
                        if rel:
                            self.state.selected_objects.add(rel.source_id)
                            self.state.selected_objects.add(rel.target_id)
                    else:
                        # For objects, select the object
                        self.state.selected_objects.add(obj_id)
            except (IndexError, ValueError):
                continue
        
        # Update object list selection
        self.objects_list.clearSelection()
        for i in range(self.objects_list.count()):
            item = self.objects_list.item(i)
            text = item.text()
            try:
                obj_id = int(text.split("ID: ")[1].rstrip(")"))
                if obj_id in self.state.selected_objects:
                    item.setSelected(True)
            except (IndexError, ValueError):
                continue
        
        # Highlight selected objects
        self._highlight_selected_objects()
    
    def _on_highlight_query_results(self):
        """Handle highlight query results button."""
        self._highlight_query_results()
    
    def _on_query_history_clicked(self, item):
        """Handle query history item click.
        
        Args:
            item: Clicked item
        """
        # Set query text
        query = item.text()
        self.query_input.setText(query)
        
        # Process query
        self._process_query_async(query)
    
    def _on_model_type_changed(self, model_type):
        """Handle model type change.
        
        Args:
            model_type: New model type
        """
        # Reinitialize query engine if needed
        if self.state.query_engine is not None:
            reply = QMessageBox.question(
                self, "Confirm", 
                "Changing model type requires reinitializing the query engine. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self._init_query_engine_async()
    
    def _on_about(self):
        """Handle about action."""
        about_text = """
        <h2>RECONTEXT: 3D Scene Reconstruction with Semantic Understanding</h2>
        <p>Version 1.0</p>
        <p>Built with:</p>
        <ul>
            <li>Open3D for 3D reconstruction and visualization</li>
            <li>PyTorch for deep learning</li>
            <li>Mask2Former for semantic segmentation</li>
            <li>CLIP for vision-language understanding</li>
        </ul>
        <p>&copy; 2024 RECONTEXT Team</p>
        """
        
        QMessageBox.about(self, "About RECONTEXT", about_text)
    
    def _on_shortcuts(self):
        """Handle keyboard shortcuts action."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
            <tr><td><b>R</b></td><td>Reset view</td></tr>
            <tr><td><b>Ctrl+O</b></td><td>Open file</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Export scene graph</td></tr>
            <tr><td><b>Ctrl+E</b></td><td>Export visualization</td></tr>
            <tr><td><b>Ctrl+Q</b></td><td>Quit</td></tr>
        </table>
        
        <h3>Mouse Controls</h3>
        <table>
            <tr><td><b>Left button</b></td><td>Rotate</td></tr>
            <tr><td><b>Right button</b></td><td>Pan</td></tr>
            <tr><td><b>Mouse wheel</b></td><td>Zoom</td></tr>
            <tr><td><b>Ctrl + Left button</b></td><td>Pick point</td></tr>
        </table>
        """
        
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)


def main():
    """Main entry point for the interactive viewer."""
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="RECONTEXT Interactive Viewer")
    parser.add_argument("--pointcloud", help="Path to point cloud file (.ply, .pcd)")
    parser.add_argument("--mesh", help="Path to mesh file (.ply, .obj)")
    parser.add_argument("--scene_graph", help="Path to scene graph file (.pkl)")
    
    args = parser.parse_args()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create viewer
    viewer = InteractiveViewer(
        pointcloud_path=args.pointcloud,
        mesh_path=args.mesh,
        scene_graph=args.scene_graph
    )
    
    # Show viewer
    viewer.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()