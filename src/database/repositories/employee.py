"""
Employee repository for database operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from .base import BaseRepository
from ..models.employee import Employee


class EmployeeRepository(BaseRepository):
    """Repository for employee data operations"""
    
    def __init__(self, db: Session):
        super().__init__(db, Employee)
    
    def get_by_employee_id(self, employee_id: str) -> Optional[Employee]:
        """Get employee by employee_id"""
        return self.db.query(Employee).filter(Employee.employee_id == employee_id).first()
    
    def get_by_department(self, department: str) -> List[Employee]:
        """Get all employees in a department"""
        return self.db.query(Employee).filter(Employee.department == department).all()
    
    def get_active_employees(self) -> List[Employee]:
        """Get all active employees"""
        return self.db.query(Employee).filter(Employee.is_active == True).all()
    
    def search_employees(
        self, 
        search_term: str = None,
        department: str = None,
        role: str = None,
        active_only: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[Employee]:
        """Search employees with multiple filters"""
        query = self.db.query(Employee)
        
        if active_only:
            query = query.filter(Employee.is_active == True)
        
        if department:
            query = query.filter(Employee.department == department)
        
        if role:
            query = query.filter(Employee.role.ilike(f'%{role}%'))
        
        if search_term:
            search_conditions = [
                Employee.employee_id.ilike(f'%{search_term}%'),
                Employee.first_name.ilike(f'%{search_term}%'),
                Employee.last_name.ilike(f'%{search_term}%'),
                Employee.department.ilike(f'%{search_term}%'),
                Employee.role.ilike(f'%{search_term}%')
            ]
            query = query.filter(or_(*search_conditions))
        
        return query.offset(skip).limit(limit).all()
    
    def get_employees_by_energy_range(
        self,
        energy_type: str,
        min_value: float,
        max_value: float
    ) -> List[Employee]:
        """Get employees with energy values in specified range"""
        if energy_type not in ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']:
            raise ValueError("Invalid energy type")
        
        energy_field = getattr(Employee, energy_type)
        return self.db.query(Employee).filter(
            and_(
                energy_field >= min_value,
                energy_field <= max_value,
                Employee.is_active == True
            )
        ).all()
    
    def get_employees_by_dominant_energy(self, energy_type: str) -> List[Employee]:
        """Get employees with specified dominant energy type"""
        if energy_type not in ['red', 'blue', 'green', 'yellow']:
            raise ValueError("Invalid energy type")
        
        energy_field = getattr(Employee, f'{energy_type}_energy')
        other_fields = [
            getattr(Employee, f'{e}_energy') 
            for e in ['red', 'blue', 'green', 'yellow'] 
            if e != energy_type
        ]
        
        conditions = [energy_field > field for field in other_fields]
        return self.db.query(Employee).filter(
            and_(
                Employee.is_active == True,
                *conditions
            )
        ).all()
    
    def get_department_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics by department"""
        result = self.db.query(
            Employee.department,
            func.count(Employee.id).label('employee_count'),
            func.avg(Employee.red_energy).label('avg_red'),
            func.avg(Employee.blue_energy).label('avg_blue'),
            func.avg(Employee.green_energy).label('avg_green'),
            func.avg(Employee.yellow_energy).label('avg_yellow')
        ).filter(
            Employee.is_active == True
        ).group_by(
            Employee.department
        ).all()
        
        return [
            {
                'department': row.department,
                'employee_count': row.employee_count,
                'avg_energies': {
                    'red_energy': round(row.avg_red, 2) if row.avg_red else 0,
                    'blue_energy': round(row.avg_blue, 2) if row.avg_blue else 0,
                    'green_energy': round(row.avg_green, 2) if row.avg_green else 0,
                    'yellow_energy': round(row.avg_yellow, 2) if row.avg_yellow else 0
                }
            }
            for row in result
        ]
    
    def validate_and_normalize_employee(self, employee: Employee) -> Dict[str, Any]:
        """Validate and normalize employee energy data"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'normalized': False
        }
        
        # Check energy sum
        total_energy = (
            employee.red_energy + employee.blue_energy + 
            employee.green_energy + employee.yellow_energy
        )
        
        if total_energy < 95 or total_energy > 105:
            validation_result['warnings'].append(
                f"Energy sum is {total_energy:.1f}%, expected ~100%"
            )
            
            # Normalize if significantly off
            if total_energy < 90 or total_energy > 110:
                employee.normalize_energies()
                validation_result['normalized'] = True
        
        # Check individual energy ranges
        for energy_type in ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']:
            value = getattr(employee, energy_type)
            if value < 0 or value > 100:
                validation_result['errors'].append(
                    f"{energy_type} value {value} is outside valid range (0-100)"
                )
                validation_result['is_valid'] = False
        
        # Check for missing required fields
        if not employee.employee_id:
            validation_result['errors'].append("Employee ID is required")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def bulk_import_employees(self, employee_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk import employees with validation"""
        results = {
            'imported': 0,
            'updated': 0,
            'errors': 0,
            'warnings': [],
            'error_details': []
        }
        
        for data in employee_data:
            try:
                # Check if employee exists
                existing = self.get_by_employee_id(data.get('employee_id'))
                
                if existing:
                    # Update existing employee
                    self.update(existing.id, data)
                    results['updated'] += 1
                else:
                    # Create new employee
                    employee = Employee(**data)
                    validation = self.validate_and_normalize_employee(employee)
                    
                    if validation['is_valid']:
                        self.db.add(employee)
                        results['imported'] += 1
                        
                        if validation['warnings']:
                            results['warnings'].extend(validation['warnings'])
                    else:
                        results['errors'] += 1
                        results['error_details'].append({
                            'employee_id': data.get('employee_id'),
                            'errors': validation['errors']
                        })
                        
            except Exception as e:
                results['errors'] += 1
                results['error_details'].append({
                    'employee_id': data.get('employee_id'),
                    'errors': [str(e)]
                })
        
        if results['imported'] > 0 or results['updated'] > 0:
            self.db.commit()
        
        return results
    
    def get_energy_distribution(self) -> Dict[str, Any]:
        """Get energy distribution statistics across all active employees"""
        result = self.db.query(
            func.avg(Employee.red_energy).label('avg_red'),
            func.avg(Employee.blue_energy).label('avg_blue'),
            func.avg(Employee.green_energy).label('avg_green'),
            func.avg(Employee.yellow_energy).label('avg_yellow'),
            func.stddev(Employee.red_energy).label('std_red'),
            func.stddev(Employee.blue_energy).label('std_blue'),
            func.stddev(Employee.green_energy).label('std_green'),
            func.stddev(Employee.yellow_energy).label('std_yellow')
        ).filter(Employee.is_active == True).first()
        
        return {
            'averages': {
                'red_energy': round(result.avg_red, 2) if result.avg_red else 0,
                'blue_energy': round(result.avg_blue, 2) if result.avg_blue else 0,
                'green_energy': round(result.avg_green, 2) if result.avg_green else 0,
                'yellow_energy': round(result.avg_yellow, 2) if result.avg_yellow else 0
            },
            'standard_deviations': {
                'red_energy': round(result.std_red, 2) if result.std_red else 0,
                'blue_energy': round(result.std_blue, 2) if result.std_blue else 0,
                'green_energy': round(result.std_green, 2) if result.std_green else 0,
                'yellow_energy': round(result.std_yellow, 2) if result.std_yellow else 0
            }
        }