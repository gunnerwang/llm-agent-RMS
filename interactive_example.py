#!/usr/bin/env python3
"""
Interactive Design Example
Demonstrates the new interactive workflow with user confirmation steps
"""

from LLM4RMS import LLM4RMS

def main():
    """Example of using the interactive design workflow"""
    
    print("🎨 LLM4RMS Interactive Workflow Example")
    print("=" * 80)
    
    # Example scenario
    scenario_name = "Collaborative Assembly Cell"
    user_input = "Design a collaborative assembly cell where operators and a UR5 cobot work together to assemble electronic components. Include parts storage, assembly stations, and quality check areas."
    no_of_objects = 5
    room_dimensions = []  # Auto-sized
    
    print(f"📋 Scenario: {scenario_name}")
    print(f"🎯 Objective: {user_input}")
    print(f"🔧 Target Objects: {no_of_objects}")
    print(f"📐 Workspace: Auto-sized")
    print()
    
    # Initialize LLM4RMS
    llm4rms = LLM4RMS(
        no_of_objects=no_of_objects,
        user_input=user_input,
        room_dimensions=room_dimensions,
        model_name="gpt-4o"
    )
    
    # Run the interactive workflow
    try:
        success = llm4rms.interactive_design_workflow()
        
        if success:
            # Ask user if they want to save the design
            print("\n💾 Would you like to save this design?")
            save_choice = input("Enter 'yes' to save, or any other key to skip: ").strip().lower()
            
            if save_choice in ['yes', 'y']:
                save_success = llm4rms.save_interactive_design(scenario_name)
                if save_success:
                    print("\n🎉 Design workflow completed and saved successfully!")
                else:
                    print("\n⚠️ Design completed but saving failed.")
            else:
                print("\n✅ Design workflow completed (not saved).")
        else:
            print("\n❌ Design workflow failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Workflow interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")

def interactive_custom_scenario():
    """Allow user to define their own scenario interactively"""
    
    print("\n🎯 Custom Interactive Design Session")
    print("=" * 80)
    
    # Get user input for custom scenario
    scenario_name = input("Enter scenario name: ").strip() or "Custom Design"
    user_input = input("Describe your manufacturing objective: ").strip()
    
    if not user_input:
        print("❌ No objective provided. Exiting.")
        return
    
    try:
        no_of_objects = int(input("How many objects/equipment pieces? (default: 5): ").strip() or "5")
    except ValueError:
        no_of_objects = 5
        print("⚠️ Invalid input, using default: 5 objects")
    
    # Ask about room dimensions
    auto_size = input("Auto-size workspace? (y/n, default: y): ").strip().lower()
    if auto_size in ['n', 'no']:
        try:
            length = float(input("Room length (m): ").strip())
            width = float(input("Room width (m): ").strip())
            height = float(input("Room height (m): ").strip())
            room_dimensions = [length, width, height]
        except ValueError:
            print("⚠️ Invalid dimensions, using auto-sizing")
            room_dimensions = []
    else:
        room_dimensions = []
    
    print(f"\n🚀 Starting design for: {scenario_name}")
    print(f"📋 Objective: {user_input}")
    print(f"🔧 Objects: {no_of_objects}")
    print(f"📐 Workspace: {'Auto-sized' if not room_dimensions else f'{room_dimensions[0]}×{room_dimensions[1]}×{room_dimensions[2]}m'}")
    print()
    
    # Initialize and run
    llm4rms = LLM4RMS(
        no_of_objects=no_of_objects,
        user_input=user_input,
        room_dimensions=room_dimensions,
        model_name="gpt-4o"
    )
    
    try:
        success = llm4rms.interactive_design_workflow()
        
        if success:
            save_choice = input("\n💾 Save this design? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                llm4rms.save_interactive_design(scenario_name)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Session interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run example scenario")
    print("2. Create custom scenario")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            interactive_custom_scenario()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")