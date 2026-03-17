"""
URDF Parser to extract robot dimensions and workspace
"""
import xml.etree.ElementTree as ET
import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional

class URDFDimensionParser:
    def __init__(self):
        self.robots_data = {}
    
    def parse_urdf_file(self, urdf_path: str) -> Dict:
        """Parse a single URDF file and extract robot dimensions"""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            robot_name = root.get('name', 'unknown')
            
            # Extract joint transformations
            joints = self._extract_joints(root)
            
            # Calculate workspace dimensions
            dimensions = self._calculate_dimensions(joints)
            
            # Extract payload if available in robot name (e.g., irb120_3_58 = 3kg, 580mm)
            payload, reach = self._extract_specs_from_name(robot_name)
            
            return {
                'name': robot_name,
                'joints': joints,
                'dimensions': dimensions,
                'payload_kg': payload,
                'reach_mm': reach
            }
            
        except Exception as e:
            print(f"Error parsing {urdf_path}: {e}")
            return None
    
    def _extract_joints(self, root) -> List[Dict]:
        """Extract joint information from URDF"""
        joints = []
        
        for joint in root.findall('.//joint[@type="revolute"]'):
            name = joint.get('name', '')
            
            # Get origin transformation
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                rpy = origin.get('rpy', '0 0 0').split()
                
                translation = [float(x) for x in xyz]
                rotation = [float(x) for x in rpy]
                
                joints.append({
                    'name': name,
                    'translation': translation,
                    'rotation': rotation
                })
        
        return joints
    
    def _calculate_dimensions(self, joints: List[Dict]) -> Dict:
        """Calculate robot workspace dimensions from joint data"""
        # Track cumulative position through kinematic chain
        current_pos = np.array([0.0, 0.0, 0.0])
        max_reach = 0.0
        max_height = 0.0
        positions = [current_pos.copy()]
        
        for joint in joints:
            # Add translation from this joint
            translation = np.array(joint['translation'])
            current_pos += translation
            positions.append(current_pos.copy())
            
            # Track maximum reach (distance from origin)
            reach = np.linalg.norm(current_pos[:2])  # X-Y plane distance
            max_reach = max(max_reach, reach)
            
            # Track maximum height
            max_height = max(max_height, current_pos[2])
        
        # Calculate bounding box
        positions = np.array(positions)
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        # Workspace dimensions (considering robot can rotate)
        workspace_radius = max_reach
        workspace_diameter = workspace_radius * 2
        
        return {
            'max_reach_m': max_reach,
            'max_height_m': max_height,
            'workspace_diameter_m': workspace_diameter,
            'bounding_box': {
                'length': max_bounds[0] - min_bounds[0],
                'width': max_bounds[1] - min_bounds[1], 
                'height': max_bounds[2] - min_bounds[2]
            },
            'footprint': {
                'length': workspace_diameter,
                'width': workspace_diameter,
                'height': max_height
            }
        }
    
    def _extract_specs_from_name(self, robot_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract payload and reach from robot name"""
        payload = None
        reach = None
        
        # Common patterns in robot names
        # Examples: irb120_3_58 (3kg, 580mm), ur10e (10kg), kr10r1100sixx (10kg, 1100mm)
        
        # ABB IRB pattern: irb<model>_<payload>_<reach>
        abb_match = re.search(r'irb\d+[a-z]*_(\d+)_(\d+)', robot_name)
        if abb_match:
            payload = float(abb_match.group(1))
            reach = float(abb_match.group(2)) / 1000  # Convert mm to m
        
        # KUKA KR pattern: kr<payload>r<reach>
        kuka_match = re.search(r'kr(\d+)r(\d+)', robot_name)
        if kuka_match:
            payload = float(kuka_match.group(1))
            reach = float(kuka_match.group(2)) / 1000  # Convert mm to m
        
        # Universal Robots pattern: ur<payload>
        ur_match = re.search(r'ur(\d+)', robot_name)
        if ur_match:
            payload = float(ur_match.group(1))
            # UR typical reaches: UR5=850mm, UR10=1300mm
            if payload == 5:
                reach = 0.85
            elif payload == 10:
                reach = 1.3
        
        return payload, reach
    
    def scan_robots_directory(self, robots_dir: str) -> Dict:
        """Scan robots directory and parse all URDF files"""
        robot_data = {}
        
        for root, dirs, files in os.walk(robots_dir):
            for file in files:
                if file.endswith('.urdf') and not 'collision' in root:
                    urdf_path = os.path.join(root, file)
                    data = self.parse_urdf_file(urdf_path)
                    if data:
                        robot_data[data['name']] = data
        
        return robot_data
    
    def generate_equipment_catalog_update(self, robot_data: Dict) -> str:
        """Generate updated equipment catalog entries with calculated dimensions"""
        catalog_entries = []
        
        for robot_name, data in robot_data.items():
            dims = data['dimensions']
            footprint = dims['footprint']
            
            # Use calculated footprint dimensions
            length = round(footprint['length'], 2)
            width = round(footprint['width'], 2) 
            height = round(footprint['height'], 2)
            
            # Create description with specs
            payload_str = f"{data['payload_kg']}kg payload, " if data['payload_kg'] else ""
            reach_str = f"{int(data['reach_mm']*1000)}mm reach" if data['reach_mm'] else f"{data['dimensions']['max_reach_m']:.1f}m reach"
            
            # Determine manufacturer
            if robot_name.startswith('abb'):
                manufacturer = "ABB"
                model_name = robot_name.replace('abb_', '').upper()
            elif robot_name.startswith('kuka'):
                manufacturer = "KUKA" 
                model_name = robot_name.replace('kuka_', '').upper()
            elif robot_name.startswith('ur'):
                manufacturer = "Universal Robots"
                model_name = robot_name.upper()
            else:
                manufacturer = "Robot"
                model_name = robot_name
            
            description = f"{manufacturer} {model_name} robot"
            if payload_str or reach_str:
                description += f", {payload_str}{reach_str}"
            
            entry = f'''    "{robot_name}": {{
        "category": "robots",
        "description": "{description}",
        "approximate_size": {{"length": {length}, "width": {width}, "height": {height}}},
        "style": "Modern",
        "material": "Metal and Electronics"
    }}'''
            catalog_entries.append(entry)
        
        return ',\n'.join(catalog_entries)

def main():
    parser = URDFDimensionParser()
    
    # Scan robots directory
    robots_dir = "/mnt/portable/tianyu/LLM4RMS/Unity/Robots"
    print("Scanning robots directory...")
    robot_data = parser.scan_robots_directory(robots_dir)
    
    print(f"Found {len(robot_data)} robots with URDF files:")
    for name, data in robot_data.items():
        dims = data['dimensions']
        print(f"{name}: {dims['footprint']['length']:.2f}x{dims['footprint']['width']:.2f}x{dims['footprint']['height']:.2f}m")
    
    # Generate catalog update
    catalog_update = parser.generate_equipment_catalog_update(robot_data)
    
    # Save to file
    with open('/mnt/portable/tianyu/LLM4RMS/Unity/robot_dimensions_calculated.txt', 'w') as f:
        f.write("# Calculated robot dimensions from URDF files\n")
        f.write(catalog_update)
    
    print("\nGenerated equipment catalog update saved to robot_dimensions_calculated.txt")

if __name__ == "__main__":
    main()