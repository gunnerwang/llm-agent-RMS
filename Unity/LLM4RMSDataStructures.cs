using System;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;

namespace LLM4RMSExporter
{
    /// <summary>
    /// Data structures matching the LLM4RMS JSON format
    /// </summary>
    
    [System.Serializable]
    public class LLM4RMSObject
    {
        public string new_object_id;
        public string style;
        public string material;
        public Size3D size_in_meters;
        public bool is_on_the_floor;
        public string facing;
        public Placement placement;
        public Rotation rotation;
        public Cluster cluster;
        public Position3D position;
        
        // For wall objects
        public string itemType;
        public Vector3 size; // Used for walls
    }
    
    [System.Serializable]
    public class Size3D
    {
        public float length;
        public float width;
        public float height;
    }
    
    [System.Serializable]
    public class Position3D
    {
        public float x;
        public float y;
        public float z;
    }
    
    [System.Serializable]
    public class Rotation
    {
        public float z_angle;
    }
    
    [System.Serializable]
    public class Placement
    {
        public List<RoomLayoutElement> room_layout_elements;
        public List<ObjectRelation> objects_in_room;
        
        public Placement()
        {
            room_layout_elements = new List<RoomLayoutElement>();
            objects_in_room = new List<ObjectRelation>();
        }
    }
    
    [System.Serializable]
    public class RoomLayoutElement
    {
        public string layout_element_id;
        public string preposition;
    }
    
    [System.Serializable]
    public class ObjectRelation
    {
        public string object_id;
        public string preposition;
        public bool is_adjacent;
    }
    
    [System.Serializable]
    public class Cluster
    {
        public ConstraintArea constraint_area;
        
        public Cluster()
        {
            constraint_area = new ConstraintArea();
        }
    }
    
    [System.Serializable]
    public class ConstraintArea
    {
        public float x_neg;
        public float x_pos;
        public float y_neg;
        public float y_pos;
    }
    
    /// <summary>
    /// Spatial relationship analyzer to determine object relationships
    /// </summary>
    public static class SpatialRelationshipAnalyzer
    {
        public static List<ObjectRelation> AnalyzeRelationships(GameObject sourceObj, List<GameObject> allObjects, float proximityThreshold = 2.0f)
        {
            List<ObjectRelation> relations = new List<ObjectRelation>();
            
            Vector3 sourcePos = sourceObj.transform.position;
            Bounds sourceBounds = GetObjectBounds(sourceObj);
            
            foreach (GameObject targetObj in allObjects)
            {
                if (targetObj == sourceObj || targetObj == null) continue;
                
                Vector3 targetPos = targetObj.transform.position;
                Bounds targetBounds = GetObjectBounds(targetObj);
                
                float distance = Vector3.Distance(sourcePos, targetPos);
                
                if (distance <= proximityThreshold)
                {
                    string relationship = DetermineRelationship(sourceBounds, targetBounds);
                    bool isAdjacent = distance <= (sourceBounds.size.magnitude + targetBounds.size.magnitude) / 2 + 0.5f;
                    
                    relations.Add(new ObjectRelation
                    {
                        object_id = targetObj.name,
                        preposition = relationship,
                        is_adjacent = isAdjacent
                    });
                }
            }
            
            return relations;
        }
        
        private static string DetermineRelationship(Bounds source, Bounds target)
        {
            Vector3 direction = target.center - source.center;
            
            // Normalize to get primary direction
            if (Mathf.Abs(direction.x) > Mathf.Abs(direction.z))
            {
                return direction.x > 0 ? "right of" : "left of";
            }
            else
            {
                return direction.z > 0 ? "in front of" : "behind";
            }
        }
        
        private static Bounds GetObjectBounds(GameObject obj)
        {
            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                return renderer.bounds;
            }
            
            Collider collider = obj.GetComponent<Collider>();
            if (collider != null)
            {
                return collider.bounds;
            }
            
            return new Bounds(obj.transform.position, Vector3.one);
        }
        
        public static string DetermineWallProximity(Vector3 objectPosition, Vector3 roomDimensions, float threshold = 1.5f)
        {
            List<string> nearWalls = new List<string>();
            
            // Check each wall
            if (objectPosition.z <= threshold)
                nearWalls.Add("south_wall");
            if (objectPosition.z >= roomDimensions.z - threshold)
                nearWalls.Add("north_wall");
            if (objectPosition.x <= threshold)
                nearWalls.Add("west_wall");
            if (objectPosition.x >= roomDimensions.x - threshold)
                nearWalls.Add("east_wall");
            
            // Return the most appropriate wall reference
            if (nearWalls.Count == 0)
                return "middle of the room";
            else if (nearWalls.Count == 1)
                return nearWalls[0];
            else
                return string.Join(" and ", nearWalls) + " corner";
        }
        
        public static string DetermineCornerPlacement(Vector3 objectPosition, Vector3 roomDimensions, float threshold = 1.5f)
        {
            bool nearSouth = objectPosition.z <= threshold;
            bool nearNorth = objectPosition.z >= roomDimensions.z - threshold;
            bool nearWest = objectPosition.x <= threshold;
            bool nearEast = objectPosition.x >= roomDimensions.x - threshold;
            
            if (nearSouth && nearWest) return "south-west corner";
            if (nearSouth && nearEast) return "south-east corner";
            if (nearNorth && nearWest) return "north-west corner";
            if (nearNorth && nearEast) return "north-east corner";
            
            return null; // Not in a corner
        }
    }
    
    /// <summary>
    /// Utility class for Unity Editor-specific GUI operations
    /// </summary>
    public static class EditorGUIUtils
    {
        public static LayerMask LayerMaskField(string label, LayerMask layerMask)
        {
            #if UNITY_EDITOR
            var layers = new List<string>();
            var layerNumbers = new List<int>();
            
            for (int i = 0; i < 32; i++)
            {
                string layerName = LayerMask.LayerToName(i);
                if (!string.IsNullOrEmpty(layerName))
                {
                    layers.Add(layerName);
                    layerNumbers.Add(i);
                }
            }
            
            int maskWithoutEmpty = 0;
            for (int i = 0; i < layerNumbers.Count; i++)
            {
                if (((1 << layerNumbers[i]) & layerMask.value) > 0)
                    maskWithoutEmpty |= (1 << i);
            }
            
            maskWithoutEmpty = UnityEditor.EditorGUILayout.MaskField(label, maskWithoutEmpty, layers.ToArray());
            
            int mask = 0;
            for (int i = 0; i < layerNumbers.Count; i++)
            {
                if ((maskWithoutEmpty & (1 << i)) > 0)
                    mask |= (1 << layerNumbers[i]);
            }
            
            layerMask.value = mask;
            return layerMask;
            #else
            return layerMask;
            #endif
        }
    }
    
    /// <summary>
    /// Scene validation and optimization utilities
    /// </summary>
    public static class SceneValidation
    {
        public static List<ValidationIssue> ValidateScene(List<GameObject> equipment, Vector3 roomDimensions)
        {
            List<ValidationIssue> issues = new List<ValidationIssue>();
            
            // Check for overlapping objects
            for (int i = 0; i < equipment.Count; i++)
            {
                for (int j = i + 1; j < equipment.Count; j++)
                {
                    if (CheckOverlap(equipment[i], equipment[j]))
                    {
                        issues.Add(new ValidationIssue
                        {
                            severity = IssueSeverity.Warning,
                            message = $"Objects {equipment[i].name} and {equipment[j].name} may be overlapping",
                            affectedObjects = new[] { equipment[i], equipment[j] }
                        });
                    }
                }
            }
            
            // Check for objects outside room bounds
            foreach (GameObject obj in equipment)
            {
                Vector3 pos = obj.transform.position;
                if (pos.x < 0 || pos.x > roomDimensions.x || pos.z < 0 || pos.z > roomDimensions.z)
                {
                    issues.Add(new ValidationIssue
                    {
                        severity = IssueSeverity.Error,
                        message = $"Object {obj.name} is outside room boundaries",
                        affectedObjects = new[] { obj }
                    });
                }
            }
            
            // Check for missing equipment types
            var uniqueTypes = new HashSet<string>();
            foreach (GameObject obj in equipment)
            {
                // This would need to be connected to the equipment mapping
                // uniqueTypes.Add(GetEquipmentType(obj.name));
            }
            
            return issues;
        }
        
        private static bool CheckOverlap(GameObject obj1, GameObject obj2)
        {
            Bounds bounds1 = GetObjectBounds(obj1);
            Bounds bounds2 = GetObjectBounds(obj2);
            
            return bounds1.Intersects(bounds2);
        }
        
        private static Bounds GetObjectBounds(GameObject obj)
        {
            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                return renderer.bounds;
            }
            
            Collider collider = obj.GetComponent<Collider>();
            if (collider != null)
            {
                return collider.bounds;
            }
            
            return new Bounds(obj.transform.position, Vector3.one);
        }
    }
    
    [System.Serializable]
    public class ValidationIssue
    {
        public IssueSeverity severity;
        public string message;
        public GameObject[] affectedObjects;
    }
    
    public enum IssueSeverity
    {
        Info,
        Warning,
        Error
    }
}