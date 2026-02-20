"""
Person 3 - Progress Tracker
Visual progress tracker for training pipeline
"""
from pathlib import Path
import json
from datetime import datetime

class ProgressTracker:
    """Track progress of Person 3's work"""
    
    def __init__(self):
        self.progress_file = Path("progress.json")
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "started": datetime.now().isoformat(),
                "tasks": {
                    "dependencies": {"status": "pending", "completed": None},
                    "datasets": {"status": "pending", "completed": None},
                    "flan_t5": {"status": "pending", "completed": None},
                    "pegasus": {"status": "pending", "completed": None},
                    "mistral": {"status": "pending", "completed": None},
                    "dipper": {"status": "pending", "completed": None},
                    "integration": {"status": "pending", "completed": None}
                }
            }
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)
    
    def update_task(self, task_name, status):
        """Update task status"""
        if task_name in self.progress["tasks"]:
            self.progress["tasks"][task_name]["status"] = status
            if status == "completed":
                self.progress["tasks"][task_name]["completed"] = datetime.now().isoformat()
            self.save_progress()
    
    def check_actual_status(self):
        """Check actual status from filesystem"""
        # Check dependencies
        try:
            import torch
            import transformers
            self.update_task("dependencies", "completed")
        except:
            pass
        
        # Check datasets
        if (Path("data/train.jsonl").exists() and 
            Path("data/validation.jsonl").exists() and 
            Path("data/test.jsonl").exists()):
            self.update_task("datasets", "completed")
        
        # Check models
        checkpoints = Path("checkpoints")
        if (checkpoints / "flan_t5_xl_final").exists():
            self.update_task("flan_t5", "completed")
        if (checkpoints / "pegasus_large_final").exists():
            self.update_task("pegasus", "completed")
        if (checkpoints / "mistral_7b_qlora_final").exists():
            self.update_task("mistral", "completed")
        if (checkpoints / "dipper_xxl").exists():
            self.update_task("dipper", "completed")
        
        # Check integration
        if Path("humanizer.py").exists():
            try:
                from humanizer import humanize
                self.update_task("integration", "completed")
            except:
                pass
    
    def display(self):
        """Display progress visually"""
        self.check_actual_status()
        
        print("=" * 80)
        print("PERSON 3 - PROGRESS TRACKER")
        print("=" * 80)
        
        tasks = {
            "dependencies": "Install Dependencies",
            "datasets": "Download Datasets",
            "flan_t5": "Train Flan-T5-XL",
            "pegasus": "Train PEGASUS-large",
            "mistral": "Train Mistral-7B (Optional)",
            "dipper": "Setup DIPPER (Optional)",
            "integration": "Integration Ready"
        }
        
        completed = 0
        total = len(tasks)
        
        print("\nTasks:")
        for task_id, task_name in tasks.items():
            task_data = self.progress["tasks"][task_id]
            status = task_data["status"]
            
            if status == "completed":
                icon = "✓"
                color = "DONE"
                completed += 1
            elif status == "in_progress":
                icon = "⋯"
                color = "IN PROGRESS"
            else:
                icon = "○"
                color = "PENDING"
            
            print(f"  {icon} {task_name:35s} [{color}]")
            
            if task_data["completed"]:
                completed_time = datetime.fromisoformat(task_data["completed"])
                print(f"     Completed: {completed_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Progress bar
        progress_pct = (completed / total) * 100
        bar_length = 50
        filled = int(bar_length * completed / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\nOverall Progress:")
        print(f"  [{bar}] {progress_pct:.0f}% ({completed}/{total} tasks)")
        
        # Estimate
        if completed > 0 and completed < total:
            started = datetime.fromisoformat(self.progress["started"])
            elapsed = (datetime.now() - started).total_seconds() / 3600  # hours
            estimated_total = (elapsed / completed) * total
            remaining = estimated_total - elapsed
            
            print(f"\nTime Estimate:")
            print(f"  Elapsed: {elapsed:.1f} hours")
            print(f"  Remaining: {remaining:.1f} hours (~{remaining/24:.1f} days)")
        
        # Next steps
        print(f"\nNext Steps:")
        for task_id, task_name in tasks.items():
            if self.progress["tasks"][task_id]["status"] == "pending":
                print(f"  → {task_name}")
                
                # Show command
                commands = {
                    "dependencies": "pip install -r requirements.txt",
                    "datasets": "python dataset_downloader.py",
                    "flan_t5": "python train_flan_t5.py",
                    "pegasus": "python train_pegasus.py",
                    "mistral": "python train_mistral.py",
                    "dipper": "python setup_dipper.py",
                    "integration": "python humanizer.py"
                }
                if task_id in commands:
                    print(f"     Command: {commands[task_id]}")
                break
        else:
            print("  ✓ All tasks completed!")
            print("  → Share with Person 4: INTEGRATION_GUIDE.md")
        
        print("=" * 80)
    
    def mark_in_progress(self, task_name):
        """Mark task as in progress"""
        self.update_task(task_name, "in_progress")
        print(f"[PROGRESS] Started: {task_name}")
    
    def mark_completed(self, task_name):
        """Mark task as completed"""
        self.update_task(task_name, "completed")
        print(f"[PROGRESS] Completed: {task_name}")

def main():
    """Display progress"""
    tracker = ProgressTracker()
    tracker.display()

if __name__ == "__main__":
    main()
