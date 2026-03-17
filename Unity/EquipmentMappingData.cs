using System;
using System.Collections.Generic;
using UnityEngine;

namespace LLM4RMSExporter
{
    [CreateAssetMenu(fileName = "EquipmentMapping", menuName = "LLM4RMS/Equipment Mapping Data")]
    public class EquipmentMappingData : ScriptableObject
    {
        [Header("Equipment Type Mappings")]
        [SerializeField]
        private List<EquipmentMapping> mappings = new List<EquipmentMapping>();
        
        [Header("Default Equipment Properties")]
        [SerializeField]
        private List<EquipmentTypeInfo> equipmentTypes = new List<EquipmentTypeInfo>();
        
        private Dictionary<string, string> nameToTypeCache;
        private Dictionary<string, EquipmentTypeInfo> typeInfoCache;
        
        private void OnEnable()
        {
            BuildCaches();
        }
        
        private void OnValidate()
        {
            BuildCaches();
        }
        
        private void BuildCaches()
        {
            nameToTypeCache = new Dictionary<string, string>();
            typeInfoCache = new Dictionary<string, EquipmentTypeInfo>();
            
            // Build name to type mapping cache
            foreach (var mapping in mappings)
            {
                foreach (string pattern in mapping.gameObjectNamePatterns)
                {
                    if (!string.IsNullOrEmpty(pattern))
                    {
                        nameToTypeCache[pattern.ToLower()] = mapping.equipmentType;
                    }
                }
            }
            
            // Build type info cache
            foreach (var typeInfo in equipmentTypes)
            {
                if (!string.IsNullOrEmpty(typeInfo.typeName))
                {
                    typeInfoCache[typeInfo.typeName] = typeInfo;
                }
            }
        }
        
        public string GetEquipmentType(string gameObjectName)
        {
            if (nameToTypeCache == null)
                BuildCaches();
                
            string lowerName = gameObjectName.ToLower();
            
            // First try exact match
            if (nameToTypeCache.ContainsKey(lowerName))
            {
                return nameToTypeCache[lowerName];
            }
            
            // Then try partial matches
            foreach (var kvp in nameToTypeCache)
            {
                if (lowerName.Contains(kvp.Key) || kvp.Key.Contains(lowerName))
                {
                    return kvp.Value;
                }
            }
            
            return "Unknown";
        }
        
        public EquipmentTypeInfo GetEquipmentInfo(string equipmentType)
        {
            if (typeInfoCache == null)
                BuildCaches();
                
            return typeInfoCache.ContainsKey(equipmentType) ? typeInfoCache[equipmentType] : null;
        }
        
        public void InitializeDefaultMappings()
        {
            mappings.Clear();
            equipmentTypes.Clear();
            
            // Initialize with LLM4RMS equipment catalog mappings
            var defaultMappings = new[]
            {
                new EquipmentMapping
                {
                    equipmentType = "Pallet",
                    gameObjectNamePatterns = new[] { "pallet", "Pallet", "staging", "storage" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Pallet_Jack",
                    gameObjectNamePatterns = new[] { "pallet_jack", "palletjack", "hand_truck", "trolley" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Conveyor_6ft",
                    gameObjectNamePatterns = new[] { "conveyor", "belt", "transport" }
                },
                new EquipmentMapping
                {
                    equipmentType = "ConveyorCorner_6ft",
                    gameObjectNamePatterns = new[] { "conveyor_corner", "belt_corner", "corner_transport" }
                },
                new EquipmentMapping
                {
                    equipmentType = "MFG_Equip_30ftx7ft_w_Exhaust",
                    gameObjectNamePatterns = new[] { "cnc", "machine", "manufacturing", "mill", "lathe" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Power_Cutter",
                    gameObjectNamePatterns = new[] { "cutter", "saw", "plasma", "laser" }
                },
                new EquipmentMapping
                {
                    equipmentType = "ScissorLift",
                    gameObjectNamePatterns = new[] { "scissor", "lift", "platform", "elevator" }
                },
                new EquipmentMapping
                {
                    equipmentType = "ShelvingRack",
                    gameObjectNamePatterns = new[] { "shelf", "rack", "storage", "shelving" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Table_6ft",
                    gameObjectNamePatterns = new[] { "table", "workbench", "desk", "surface" }
                },
                new EquipmentMapping
                {
                    equipmentType = "ur10e_robot",
                    gameObjectNamePatterns = new[] { "ur10", "universal_robot", "cobot", "robot_arm" }
                },
                new EquipmentMapping
                {
                    equipmentType = "abb_irb2600_12_165",
                    gameObjectNamePatterns = new[] { "abb", "irb2600", "industrial_robot", "robot" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Camera_Stand_7ft",
                    gameObjectNamePatterns = new[] { "camera", "vision", "inspection", "qc" }
                },
                new EquipmentMapping
                {
                    equipmentType = "VentilatorFan_Straight",
                    gameObjectNamePatterns = new[] { "fan", "ventilator", "exhaust", "air" }
                },
                new EquipmentMapping
                {
                    equipmentType = "SafetyRailing_8ft",
                    gameObjectNamePatterns = new[] { "railing", "fence", "barrier", "safety" }
                },
                new EquipmentMapping
                {
                    equipmentType = "Air_Compressor",
                    gameObjectNamePatterns = new[] { "compressor", "air_supply", "pneumatic" }
                }
            };
            
            mappings.AddRange(defaultMappings);
            
            // Initialize default equipment type info based on LLM4RMS catalog
            var defaultTypeInfo = new[]
            {
                new EquipmentTypeInfo
                {
                    typeName = "Pallet",
                    category = "material_flow",
                    style = "Industrial",
                    material = "Wood",
                    description = "Standard wooden pallet for staging inbound or outbound goods"
                },
                new EquipmentTypeInfo
                {
                    typeName = "Pallet_Jack",
                    category = "material_flow",
                    style = "Industrial",
                    material = "Metal",
                    description = "Manual pallet jack for repositioning pallets between workzones"
                },
                new EquipmentTypeInfo
                {
                    typeName = "Conveyor_6ft",
                    category = "material_flow",
                    style = "Industrial",
                    material = "Metal",
                    description = "6 foot straight conveyor section for linear transport runs"
                },
                new EquipmentTypeInfo
                {
                    typeName = "ConveyorCorner_6ft",
                    category = "material_flow",
                    style = "Industrial",
                    material = "Metal",
                    description = "6 foot corner conveyor section for 90-degree directional changes"
                },
                new EquipmentTypeInfo
                {
                    typeName = "MFG_Equip_30ftx7ft_w_Exhaust",
                    category = "machining_fabrication",
                    style = "Industrial",
                    material = "Metal",
                    description = "Large manufacturing equipment with integrated exhaust system"
                },
                new EquipmentTypeInfo
                {
                    typeName = "ur10e_robot",
                    category = "assembly_collaboration",
                    style = "Industrial",
                    material = "Metal",
                    description = "Universal Robots UR10e collaborative robot arm"
                },
                new EquipmentTypeInfo
                {
                    typeName = "abb_irb2600_12_165",
                    category = "assembly_collaboration",
                    style = "Industrial",
                    material = "Metal",
                    description = "ABB IRB 2600 industrial robot with 12kg payload and 1.65m reach"
                },
                new EquipmentTypeInfo
                {
                    typeName = "Table_6ft",
                    category = "assembly_collaboration",
                    style = "Industrial",
                    material = "Metal",
                    description = "Six-foot industrial work table for assembly and inspection tasks"
                },
                new EquipmentTypeInfo
                {
                    typeName = "Camera_Stand_7ft",
                    category = "quality_finishing",
                    style = "Industrial",
                    material = "Metal",
                    description = "Seven-foot camera stand for visual inspection and quality control"
                },
                new EquipmentTypeInfo
                {
                    typeName = "Air_Compressor",
                    category = "utilities",
                    style = "Industrial",
                    material = "Metal",
                    description = "Industrial air compressor for pneumatic tools and systems"
                }
            };
            
            equipmentTypes.AddRange(defaultTypeInfo);
            
            BuildCaches();
        }
        
        public void AddMapping(string equipmentType, params string[] namePatterns)
        {
            var mapping = new EquipmentMapping
            {
                equipmentType = equipmentType,
                gameObjectNamePatterns = namePatterns
            };
            
            mappings.Add(mapping);
            BuildCaches();
        }
        
        public void RemoveMapping(string equipmentType)
        {
            mappings.RemoveAll(m => m.equipmentType == equipmentType);
            BuildCaches();
        }
        
        public List<string> GetAllEquipmentTypes()
        {
            List<string> types = new List<string>();
            foreach (var mapping in mappings)
            {
                if (!types.Contains(mapping.equipmentType))
                {
                    types.Add(mapping.equipmentType);
                }
            }
            return types;
        }
        
        public List<string> GetPatternsForType(string equipmentType)
        {
            foreach (var mapping in mappings)
            {
                if (mapping.equipmentType == equipmentType)
                {
                    return new List<string>(mapping.gameObjectNamePatterns);
                }
            }
            return new List<string>();
        }
    }
    
    [System.Serializable]
    public class EquipmentMapping
    {
        [Header("Equipment Type")]
        public string equipmentType;
        
        [Header("GameObject Name Patterns")]
        [Tooltip("GameObject names (or partial names) that map to this equipment type")]
        public string[] gameObjectNamePatterns;
    }
    
    [System.Serializable]
    public class EquipmentTypeInfo
    {
        [Header("Basic Info")]
        public string typeName;
        public string category;
        public string style = "Industrial";
        public string material = "Metal";
        
        [Header("Description")]
        [TextArea(2, 4)]
        public string description;
        
        [Header("Default Dimensions (meters)")]
        public Vector3 defaultSize = Vector3.one;
        
        [Header("Behavioral Properties")]
        public bool isMovable = true;
        public bool requiresFloorSpace = true;
        public bool requiresPower = false;
        public bool requiresAir = false;
        
        [Header("Process Stage")]
        [Tooltip("Which manufacturing process stage this equipment belongs to")]
        public ProcessStage processStage = ProcessStage.Assembly;
    }
    
    public enum ProcessStage
    {
        MaterialFlow,
        MachiningFabrication,
        Assembly,
        QualityFinishing,
        Utilities
    }
}